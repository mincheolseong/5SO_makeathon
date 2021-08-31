#Agent
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D

import numpy as np


def print_node(x): #예 : print_node(3) -> 3 / 3 로 3이 2번 나옴
    print(x)       #예 : print_node('3') -> 3 / '3'
    return x


class DDQNAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 2
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

        # Training Params
        self.learning_rate = 3e-5
        self.alpha = 0.005
        self.gamma = 0.95 #discount factor

        # Exploration strategy
        self.soft_max_scaling = 0.1

        # Global-Local Map
        self.global_map_scaling = 3 #높일수록 trainable parameters 줄어듦
        self.local_map_size = 17

        # Scalar inputs instead of map
        self.use_scalar_input = False
        self.relative_scalars = False
        self.blind_agent = False
        self.max_uavs = 3
        self.max_devices = 10

        # Printing
        self.print_summary = False


class DDQNAgent(object):

    def __init__(self, params: DDQNAgentParams, example_state, example_action, stats=None):

        self.params = params
        gamma = tf.constant(self.params.gamma, dtype=float) #상수값 지정
        self.align_counter = 0

        self.boolean_map_shape = example_state.get_boolean_map_shape()
        #padded_red, padded_rest를 합친 것의 shape(boolean 표현)
        self.float_map_shape = example_state.get_float_map_shape()
        #padded_red, padded_rest를 합친 것의 shape(float 표현)
        self.scalars = example_state.get_num_scalars(give_position=self.params.use_scalar_input)
        self.num_actions = len(type(example_action)) # action 총 갯수
        self.num_map_channels = self.boolean_map_shape[2] + self.float_map_shape[2]
        # 총 채널 수 : stateUtil의 pad_centered봐야 이해됨

        # Create shared inputs
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        q_star_input = Input(shape=(), name='q_star_input', dtype=tf.float32)

        if self.params.blind_agent:
            scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
            states = [scalars_input]
            self.q_network = self.build_blind_model(scalars_input)
            self.target_network = self.build_blind_model(scalars_input, 'target_')
            self.hard_update()

        elif self.params.use_scalar_input:
            devices_input = Input(shape=(3 * self.params.max_devices,), name='devices_input', dtype=tf.float32)
            uavs_input = Input(shape=(4 * self.params.max_uavs,), name='uavs_input', dtype=tf.float32)
            scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
            states = [devices_input,
                      uavs_input,
                      scalars_input]

            self.q_network = self.build_scalars_model(states)
            self.target_network = self.build_scalars_model(states, 'target_')
            self.hard_update()

        else: #input = map은 얘만 보면 됨(boolean, float, scalar를 다같이 state로 넣네)
            boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
            float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
            scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
            states = [boolean_map_input,
                      float_map_input,
                      scalars_input]

            map_cast = tf.cast(boolean_map_input, dtype=tf.float32)
            #Boolean은 0 또는 1로 바꿔 줌, float을 정수형으로 바꿔 줌
            padded_map = tf.concat([map_cast, float_map_input], axis=3)
            #z축을 기준으로 행렬matrix를 합침

            self.q_network = self.build_model(padded_map, scalars_input, states)
            self.target_network = self.build_model(padded_map, scalars_input, states, 'target_')
            self.hard_update()

            self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                          outputs=self.global_map)
            self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.local_map)
            self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.total_map)

        q_values = self.q_network.output
        q_target_values = self.target_network.output

        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        max_action = tf.argmax(q_values, axis=1, name='max_action', output_type=tf.int64)
        max_action_target = tf.argmax(q_target_values, axis=1, name='max_action', output_type=tf.int64)
        one_hot_max_action = tf.one_hot(max_action, depth=self.num_actions, dtype=float)
        q_star = tf.reduce_sum(tf.multiply(one_hot_max_action, q_target_values, name='mul_hot_target'), axis=1,
                               name='q_star')
        #tf.reduce_sum : axis = 1 이므로 행 단위로 sum
        self.q_star_model = Model(inputs=states, outputs=q_star)

        # Define Bellman loss
        one_hot_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=1.0, off_value=0.0, dtype=float)
        one_cold_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=0.0, off_value=1.0, dtype=float)
        q_old = tf.stop_gradient(tf.multiply(q_values, one_cold_rm_action))
        gamma_terminated = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), gamma)
        q_update = tf.expand_dims(tf.add(reward_input, tf.multiply(q_star_input, gamma_terminated)), 1)
        q_update_hot = tf.multiply(q_update, one_hot_rm_action)
        q_new = tf.add(q_update_hot, q_old)
        q_loss = tf.losses.MeanSquaredError()(q_new, q_values)
        self.q_loss_model = Model(
            inputs=states + [action_input, reward_input, termination_input, q_star_input],
            outputs=q_loss)

        # Exploit act model
        self.exploit_model = Model(inputs=states, outputs=max_action)
        self.exploit_model_target = Model(inputs=states, outputs=max_action_target)

        # Softmax explore model
        softmax_scaling = d(q_values, tf.constant(self.params.soft_max_scaling, dtype=float))
        softmax_action = tf.math.softmax(softmax_scaling, name='softmax_action')
        self.soft_explore_model = Model(inputs=states, outputs=softmax_action)

        self.q_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.q_loss_model.summary()

        if stats:
            stats.set_model(self.target_network)

    def build_model(self, map_proc, states_proc, inputs, name=''):

        flatten_map = self.create_map_proc(map_proc, name)

        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu',
                          name=name + 'hidden_layer_all_'+ str(k))(layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model

    def build_scalars_model(self, inputs, name=''):

        layer = Concatenate(name=name + 'concat')(inputs)
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu',
                          name=name + 'hidden_layer_all_' + str(k))(layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model

    def build_blind_model(self, inputs, name=''):

        layer = inputs
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu',
                          name=name + 'hidden_layer_all_' + str(k))(layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model

    def create_map_proc(self, conv_in, name):

        # Forking for global and local map
        # Global Map
        global_map = tf.stop_gradient(#네트워크의 특정 파트만 학습
            AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))
            #pool_size = (3,3)으로 average pooling
            (conv_in))
        #AvgPool2D(conv_in) = stop_gradient의 input이 됨

        self.global_map = global_map
        self.total_map = conv_in

        for k in range(self.params.conv_layers):
            global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                strides=(1, 1),
                                name=name + 'global_conv_' + str(k + 1))(global_map)

        flatten_global = Flatten(name=name + 'global_flatten')(global_map)

        # Local Map
        crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
        local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
        #이미지 중앙 영역 자르기(image, central_faction)
        self.local_map = local_map

        for k in range(self.params.conv_layers):
            local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                               strides=(1, 1),
                               name=name + 'local_conv_' + str(k + 1))(local_map)

        flatten_local = Flatten(name=name + 'local_flatten')(local_map)

        return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])

    def act(self, state):
        return self.get_soft_max_exploration(state)

    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    def get_exploitation_action(self, state):

        if self.params.blind_agent:
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            return self.exploit_model(scalars).numpy()[0]

        if self.params.use_scalar_input:
            devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[tf.newaxis, ...]
            uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[tf.newaxis, ...]
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            return self.exploit_model([devices_in, uavs_in, scalars]).numpy()[0]

        #이것만 보면 된다.
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def get_soft_max_exploration(self, state):

        if self.params.blind_agent:
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            p = self.soft_explore_model(scalars).numpy()[0]
        elif self.params.use_scalar_input:
            devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[tf.newaxis, ...]
            uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[tf.newaxis, ...]
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            p = self.soft_explore_model([devices_in, uavs_in, scalars]).numpy()[0]
        else:
            boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
            float_map_in = state.get_float_map()[tf.newaxis, ...]
            scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
            p = self.soft_explore_model([boolean_map_in, float_map_in, scalars]).numpy()[0]

        return np.random.choice(range(self.num_actions), size=1, p=p)

    def get_exploitation_action_target(self, state):

        if self.params.blind_agent:
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            return self.exploit_model_target(scalars).numpy()[0]

        if self.params.use_scalar_input:
            devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[tf.newaxis, ...]
            uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[tf.newaxis, ...]
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]

            return self.exploit_model_target([devices_in, uavs_in, scalars]).numpy()[0]

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model_target([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def hard_update(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def soft_update(self, alpha):
        weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        self.target_network.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights, target_weights)])

    def train(self, experiences):
        boolean_map = experiences[0]
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]

        if self.params.blind_agent:
            q_star = self.q_star_model(
                [next_scalars])
        else:
            q_star = self.q_star_model(
                [next_boolean_map, next_float_map, next_scalars])

        # Train Value network
        with tf.GradientTape() as tape:
            #context안에 실행된 모든 연산을 tape에 기록

            if self.params.blind_agent:
                q_loss = self.q_loss_model(
                    [scalars, action, reward,
                     terminated, q_star])
            else:
                q_loss = self.q_loss_model(
                    [boolean_map, float_map, scalars, action, reward,
                     terminated, q_star])
        q_grads = tape.gradient(q_loss, self.q_network.trainable_variables)
        #위의 with tf.GradientTape로 연산 과정들을 loss에 대한 q_network.trainable_variables의 미분 실행
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_network.trainable_variables))
        #gradient 직접 조작하기

        self.soft_update(self.params.alpha)

    def save_weights(self, path_to_weights):
        self.target_network.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.target_network.save(path_to_model)

    def load_weights(self, path_to_weights):
        self.q_network.load_weights(path_to_weights)
        self.hard_update()

    def get_global_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...] #차원 변경(추가하고 싶은 위치에 tf.newaxis)
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.global_map_model([boolean_map_in, float_map_in]).numpy()
    #numpy() : 텐서를 넘파이 배열로 변환

    def get_local_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.local_map_model([boolean_map_in, float_map_in]).numpy()

    def get_total_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.total_map_model([boolean_map_in, float_map_in]).numpy()

#BaseDisplay
import import_ipynb
from Map import Map
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb
from matplotlib import patches


class BaseDisplay:
    def __init__(self):
        self.arrow_scale = 14
        self.marker_size = 15

    # Display.ipynb에서 나옴 : ax=ax_traj, env_map=env_map, value_map=value_map
    # value_map = np.ones(env_map.get_size(), dtype=float)
    def create_grid_image(self, ax, env_map: Map, value_map, green=None):
        area_y_max, area_x_max = env_map.get_size()  # (140, 140)

        if green is None:
            green = np.zeros((area_y_max, area_x_max))  # 140 x 140의 영행렬

        nfz = np.expand_dims(env_map.nfz, -1)
        # rgb2hsv쓰려면 마지막 차원이 3이되야해서 어쩔 수 없이 삽입
        lz = np.expand_dims(env_map.start_land_zone, -1)
        green = np.expand_dims(green, -1)

        neither = np.logical_not(np.logical_or(np.logical_or(nfz, lz), green))  # neither = nfz, lz, green8이 아닌 곳
        # neither = nfz, lz, green이 아닌 곳

        base = np.zeros((area_y_max, area_x_max, 3))  # (140 x 140 x 3)

        nfz_color = base.copy()
        nfz_color[..., 0] = 0.8  # (140 x 140 x 0)을 0.8로

        lz_color = base.copy()
        lz_color[..., 2] = 0.8  # (140 x 140 x 1)을 0.8로

        green_color = base.copy()
        green_color[..., 1] = 0.8  # (140 x 140 x 2)을 0.8로

        neither_color = np.ones((area_y_max, area_x_max, 3), dtype=np.float)
        grid_image = green_color * green + nfz_color * nfz + lz_color * lz + neither_color * neither
        # 즉, (50 x 50 x 3)의 속행렬 (50 x 3)에서 0, 1, 2 축에 각각 nfz, green, lz를 할당한 다음 neither를 전체적으로 다 더해주는 느낌

        # value_map = final_state.coverage * 1.0 + (~final_state.coverage) * 0.75

        hsv_image = rgb2hsv(grid_image)  # 컬러맵을 hsv(색상hue, 채도saturation, 명도value)로 나타냄
        hsv_image[..., 2] *= value_map.astype('float32')
        grid_image = hsv2rgb(hsv_image)

        if (area_x_max, area_y_max) == (64, 64):
            tick_labels_x = np.arange(0, area_x_max, 4)
            tick_labels_y = np.arange(0, area_y_max, 4)
            self.arrow_scale = 14
            self.marker_size = 6
        else:
            tick_labels_x = np.arange(0, area_x_max, 10)
            tick_labels_y = np.arange(0, area_y_max, 10)
            self.arrow_scale = 14
            self.marker_size = 4

        plt.sca(ax)  # 현재 Axes를 ax로 설정
        plt.gca().set_aspect('equal', adjustable='box')  # 현재 Axes 구하기
        plt.xticks(tick_labels_x)
        plt.yticks(tick_labels_y)
        plt.axis([0, area_x_max, area_y_max, 0])
        ax.imshow(grid_image.astype(float), extent=[0, area_x_max, area_y_max, 0])
        # plt.axis('off')

        #    obst = env_map.obstacles #장애물 설정
        #    for i in range(area_x_max):
        #        for j in range(area_y_max):
        #            if obst[j, i]:
        #                rect = patches.Rectangle((i, j), 1, 1, fill=None, hatch='////', edgecolor="Black")
        #                ax.add_patch(rect)

        # offset to shift tick labels
        locs, labels = plt.xticks()
        locs_new = [x + 0.5 for x in locs]
        plt.xticks(locs_new, tick_labels_x)

        locs, labels = plt.yticks()
        locs_new = [x + 0.5 for x in locs]
        plt.yticks(locs_new, tick_labels_y)

        # ModelStats보면 trajectory를 []로 만들어 .append method를 활용해 exp 저장

    def draw_start_and_end(self, trajectory):
        for exp in trajectory:
            state, action, reward, next_state = exp

            # Identify first moves
            # 남아있는 budget이 초기 설정값이다. 즉, first move
            if state.movement_budget == state.initial_movement_budget:
                plt.scatter(state.position[0] + 0.5, state.position[1] + 0.5,
                            s=self.marker_size, marker="D", color="w")
                # "D" means diamond, "w" means white

            # Identify last moves
            if next_state.terminal:
                if next_state.landed:
                    plt.scatter(next_state.position[0] + 0.5, next_state.position[1] + 0.5,
                                s=self.marker_size, marker="D", color="green")
                else:
                    plt.scatter(next_state.position[0] + 0.5, next_state.position[1] + 0.5,
                                s=self.marker_size, marker="D", color="r")

    def draw_movement(self, from_position, to_position, color):
        y = from_position[1]
        x = from_position[0]
        dir_y = to_position[1] - y  # y축상 거리
        dir_x = to_position[0] - x  # x축상 거리
        if dir_x == 0 and dir_y == 0:  # 이동이 없다 = hover
            plt.scatter(x + 0.5, y + 0.5, marker="X", color=color)
        else:
            if abs(dir_x) >= 1 or abs(dir_y) >= 1:
                plt.quiver(x + 0.5, y + 0.5, dir_x, -dir_y, color=color,
                           scale=self.arrow_scale, scale_units='inches')
                # 처음 2개의 좌표는 화살표의 위치, 뒤 2개의 dir은 화살표의 벡터
            else:
                plt.quiver(x + 0.5, y + 0.5, dir_x, -dir_y, color=color,
                           scale=self.arrow_scale, scale_units='inches')

#BaseGrid
import import_ipynb
from ModelStats import ModelStats
import Map as Map


class BaseGridParams:
    def __init__(self):
        self.movement_range = (100, 200)
        self.map_path = 'res/Grid_image.png'


class BaseGrid:
    def __init__(self, params: BaseGridParams, stats: ModelStats):
        self.map_image = Map.load_map(params.map_path) #해당 경로path로 imread
        self.shape = self.map_image.start_land_zone.shape #imread한 map_image에 대해서 boolean값의 크기 리턴
        self.starting_vector = self.map_image.get_starting_vector() #start, land zone에 대한 좌표값 리턴
        stats.set_env_map_callback(self.get_map_image)
        #env_map_callback(최초 None값)을 get_map_image로 리턴 ??

    def get_map_image(self):
        return self.map_image

    def get_grid_size(self):
        return self.shape

    def get_no_fly(self):
        return self.map_image.nfz

    def get_landing_zone(self):
        return self.map_image.start_land_zone

#BaseState
import import_ipynb
from Map import Map


class BaseState:
    def __init__(self, map_init: Map):
        self.no_fly_zone = map_init.nfz
        self.obstacles = map_init.obstacles
        self.landing_zone = map_init.start_land_zone

    @property #private이라는 접근 제어자(__로 표현 in python)에 있는 속성값을 보다 간편하게 가져오기 위한 데코레이터
    def shape(self):
        return self.landing_zone.shape[:2]

#Channel
import numpy as np
import import_ipynb
from Shadowing import load_or_create_shadowing


class ChannelParams:
    def __init__(self):
        self.cell_edge_snr = -25  # in dB
        self.los_path_loss_exp = 2.27
        self.nlos_path_loss_exp = 3.64
        self.uav_altitude = 10.0  # in m
        self.cell_size = 10.0  # in m #cell size = 10m는 나의 Grid size = 140 x 140 임을 고려하면 너무 크지 않나
        self.los_shadowing_variance = 2.0
        self.nlos_shadowing_variance = 5.0
        self.map_path = "res/Grid_image.png"


class Channel:
    def __init__(self, params: ChannelParams):
        self.params = params
        self._norm_distance = None
        self.los_norm_factor = None
        self.los_shadowing_sigma = None
        self.nlos_shadowing_sigma = None
        self.total_shadow_map = load_or_create_shadowing(self.params.map_path)
        #말 그대로 맵에서 전체 shadowing되는 부분을 다 나타냄

    def reset(self, area_size):
        self._norm_distance = np.sqrt(2) * 0.5 * area_size * self.params.cell_size
        self.los_norm_factor = 10 ** (self.params.cell_edge_snr / 10) / (
                self._norm_distance ** (-self.params.los_path_loss_exp))
        self.los_shadowing_sigma = np.sqrt(self.params.los_shadowing_variance)
        self.nlos_shadowing_sigma = np.sqrt(self.params.nlos_shadowing_variance)

    def get_max_rate(self):
        dist = self.params.uav_altitude

        snr = self.los_norm_factor * dist ** (-self.params.los_path_loss_exp)

        rate = np.log2(1 + snr)

        return rate

    def compute_rate(self, uav_pos, device_pos):
        dist = np.sqrt(
            ((device_pos[0] - uav_pos[0]) * self.params.cell_size) ** 2 +
            ((device_pos[1] - uav_pos[1]) * self.params.cell_size) ** 2 +
            self.params.uav_altitude ** 2)

        #if self.total_shadow_map[int(round(device_pos[1])), int(round(device_pos[0])),
        #                           int(round(uav_pos[1])), int(round(uav_pos[0]))]:
        #    #UAV의 평면좌표와 device의 평면좌표 사이에 shadowing이 있다면
        #    snr = self.los_norm_factor * dist ** (
        #        -self.params.nlos_path_loss_exp) * 10 ** (np.random.normal(0., self.nlos_shadowing_sigma) / 10)
        #    print("nLoS..")
        #나의 환경에서는 LoS만 있기 때문에 얘만 사용하면 될듯함
        snr = self.los_norm_factor * dist ** (
                -self.params.los_path_loss_exp) * 10 ** (np.random.normal(0., self.los_shadowing_sigma) / 10)
        #print("LoS!!")

        rate = np.log2(1 + snr)

        return rate

#DeviceManager
import os

import numpy as np
import import_ipynb
from IoTDevice import IoTDeviceParams, DeviceList

ColorMap = ["orange", "blue", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]


class DeviceManagerParams:
    def __init__(self):
        # self.device_count_range = (2, 5)
        # self.device_count_range = (25, 25)
        self.device_count_range = 25
        self.data_range = (5.0, 20.0)
        # self.fixed_devices = False
        self.fixed_devices = False
        self.devices = IoTDeviceParams()


class DeviceManager:
    """
    The DeviceManager is used to generate DeviceLists according to DeviceManagerParams
    """

    def __init__(self, params: DeviceManagerParams):
        self.params = params

        # 디바이스 좌표 정하려면 positions_vector 무조건 알야아함
        # positions_vector : Grid의 self.device_positions

    #     free_space = np.logical_not(
    #         np.logical_or(self.map_image.obstacles, self.map_image.start_land_zone))
    #     free_idcs = np.where(free_space)
    #     self.device_positions = list(zip(free_idcs[1], free_idcs[0]))
    #     즉, 장애물, land 빼고는 device가 다 생길 수가  있다.
    def generate_device_list(self, positions_vector):
        if self.params.fixed_devices:
            return DeviceList(self.params.devices)

        ## Roll number of devices : 25
        #      device_count = np.random.randint(self.params.device_count_range[0],
        #                                       self.params.device_count_range[1] + 1)
        device_count = self.params.device_count_range
        # The number of devices : 25개로 고정
        # device_count = self.params.device_count_range

        # Roll Positions
        # Grid에서 positions_vector = self.device_positions
        #      position_idcs = np.random.choice(range(len(positions_vector)), device_count, replace=False)
        #      positions = [positions_vector[idx] for idx in position_idcs]
        positions = positions_vector  # 25 x 1 인데 튜플의 형태

        # Roll Data : 25 x 1
        datas = np.random.uniform(self.params.data_range[0], self.params.data_range[1], device_count)
        # 5 ~ 20까지 균일하게 float 형태로 뽑음

        return self.generate_device_list_from_args(device_count, positions, datas)

    def generate_device_list_from_args(self, device_count, positions, datas):
        # get colors
        # colors = ColorMap[0:max(device_count, len(ColorMap))]
        # colors = ColorMap[0:min(device_count, len(ColorMap))]
        colors = ColorMap[0]

        params = [IoTDeviceParams(position=positions[k],
                                  # color=colors[k % len(ColorMap)],
                                  data=datas[k],
                                  color=colors
                                  )
                  # for k in range(device_count)]
                  for k in range(device_count[0])]

        return DeviceList(params)

#Display
import numpy as np
import matplotlib.pyplot as plt
import io
import tensorflow as tf
import matplotlib.patches as patches

import import_ipynb
from Map import Map
from BaseDisplay import BaseDisplay


class DHDisplay(BaseDisplay):

    def __init__(self):
        super().__init__()
        self.channel = None

    def set_channel(self, channel):
        self.channel = channel

    def display_episode(self, env_map: Map, trajectory, plot=False, save_path=None):

        first_state = trajectory[0][0]
        final_state = trajectory[-1][3]

        fig_size = 5.5
        fig, ax = plt.subplots(1, 2, figsize=[2 * fig_size, fig_size])
        ax_traj = ax[0]
        ax_bar = ax[1]

        ##IoTDevice.ipynb에 DeviceList 클래스에 있는 함수들임
        #num_devices : the number of devices
        value_step = 0.4 / first_state.device_list.num_devices
        # Start with value of 200
        value_map = np.ones(env_map.get_size(), dtype=float)
        for device in first_state.device_list.get_devices():
            value_map -= value_step * self.channel.total_shadow_map[device.position[1], device.position[0]]
            #맵에서 shadowing 되는 부분에 대해서는 value 조정(but 나의 환경에서는 해당없음)

        self.create_grid_image(ax=ax_traj, env_map=env_map, value_map=value_map)

        for device in first_state.device_list.get_devices():
            ax_traj.add_patch(
                patches.Circle(np.array(device.position) + np.array((0.5, 0.5)), 0.4, facecolor=device.color,
                               edgecolor="black"))

        self.draw_start_and_end(trajectory)

        for exp in trajectory:
            idx = exp[3].device_coms[exp[0].active_agent]
            if idx == -1:
                color = "black"
            else:
                color = exp[0].device_list.devices[idx].color

            self.draw_movement(exp[0].position, exp[3].position, color=color)

        # Add bar plots
        device_list = final_state.device_list
        devices = device_list.get_devices()
        colors = [device.color for device in devices]
        names = ["total"] + colors
        colors = ["black"] + colors
        datas = [device_list.get_total_data()] + [device.data for device in devices]
        collected_datas = [device_list.get_collected_data()] + [device.collected_data for device in devices]
        y_pos = np.arange(len(colors))

        plt.sca(ax_bar)
        ax_bar.barh(y_pos, datas)
        ax_bar.barh(y_pos, collected_datas)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(names)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Data")
        ax_bar.set_aspect(- np.diff(ax_bar.get_xlim())[0] / np.diff(ax_bar.get_ylim())[0])

        # save image and return
        if save_path is not None:
            # save just the trajectory subplot 0
            extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent.x0 -= 0.3
            extent.y0 -= 0.1
            fig.savefig(save_path, bbox_inches=extent,
                        format='png', dpi=300, pad_inches=1)
        if plot:
            plt.show()

        # plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=180, bbox_inches='tight')
        buf.seek(0)
        # plt.close(fig=fig)
        plt.close('all')
        combined_image = tf.image.decode_png(buf.getvalue(), channels=3)
        combined_image = tf.expand_dims(combined_image, 0)

        return combined_image

#Environment
import copy
import tqdm
import distutils.util

import import_ipynb
from ModelStats import ModelStatsParams, ModelStats
from BaseDisplay import BaseDisplay


class BaseEnvironmentParams:
    def __init__(self):
        self.model_stats_params = ModelStatsParams()


class BaseEnvironment:
    def __init__(self, params: BaseEnvironmentParams, display: BaseDisplay):
        self.stats = ModelStats(params.model_stats_params, display=display)
        self.trainer = None
        self.grid = None
        self.rewards = None
        self.physics = None
        self.display = display
        self.episode_count = 0
        self.step_count = 0

    def fill_replay_memory(self):

        while self.trainer.should_fill_replay_memory():

            state = copy.deepcopy(self.init_episode())
            # init_episode를 deepcopy(내부 객체까지 copy해서 그 값을 계속 저장)
            while not state.terminal:
                next_state = self.step(state, random=self.trainer.params.rm_pre_fill_random)
                state = copy.deepcopy(next_state)

    def train_episode(self):
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        while not state.is_terminal():
            state = self.step(state)
            self.trainer.train_agent()

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_training_data(step=self.step_count)

        self.episode_count += 1

    def run(self):

        self.fill_replay_memory()

        print('Running ', self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.trainer.params.num_steps:
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode()

            if self.episode_count % self.trainer.params.eval_period == 0:
                self.test_episode()

            self.stats.save_if_best()

        self.stats.training_ended()

    def step(self, state, random=False):
        pass

    def init_episode(self, init_state=None):
        if init_state:
            state = copy.deepcopy(self.grid.init_scenario(init_state))
        else:
            state = copy.deepcopy(self.grid.init_episode())

        self.rewards.reset()
        self.physics.reset(state)
        return state

    def test_episode(self):
        pass

    def test_scenario(self, scenario):
        pass

    def eval(self, episodes, show=False):
        for _ in tqdm.tqdm(range(episodes)):
            self.test_episode()
            self.step_count += 1  # Increase step count so that logging works properly

            if show:
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

                resp = input('Save run? [y/N]\n')
                try:
                    if distutils.util.strtobool(resp):
                        # converting a string representation of truth to true(1) or false(0)
                        # 저장하겠다
                        save_as = input('Save as: [run_' + str(self.step_count) + ']\n')
                        if save_as == '':
                            save_as = 'run_' + str(self.step_count)
                        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                                     save_path=save_as + '.png')
                        self.stats.save_episode(save_as)
                        print("Saved as run_" + str(self.step_count))
                except ValueError:
                    pass
                print("next then")

    def eval_scenario(self, init_state):
        self.test_scenario(init_state)

        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

        resp = input('Save run? [y/N]\n')
        try:
            if distutils.util.strtobool(resp):
                save_as = input('Save as: [scenario]\n')
                if save_as == '':
                    save_as = 'scenario'
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                             save_path=save_as + '.png')
                self.stats.save_episode(save_as)
                print("Saved as", save_as)
        except ValueError:
            pass

#Grid
import numpy as np
import import_ipynb

from DeviceManager import DeviceManagerParams, DeviceManager
from State import State
from BaseGrid import BaseGrid, BaseGridParams


class GridParams(BaseGridParams):
    def __init__(self):
        super().__init__()
        self.num_agents_range = [1, 3]
        self.device_manager = DeviceManagerParams()
        self.multi_agent = False
        self.fixed_starting_idcs = False  # starting vector에서는 아무 곳에서나 출발 가능
        self.starting_idcs = [1, 2, 3]  #
        self.x_loc_device = [20, 40, 60, 80, 100]
        self.y_loc_device = [30, 50, 70, 90, 110]


class Grid(BaseGrid):

    def __init__(self, params: GridParams, stats):
        super().__init__(params, stats)
        self.params = params
        if params.multi_agent:
            self.num_agents = params.num_agents_range[0]  # 이상한데..?
        else:
            self.num_agents = 1

        self.device_list = None
        self.device_manager = DeviceManager(self.params.device_manager)

        #       free_space = np.logical_not(
        #           np.logical_or(self.map_image.obstacles, self.map_image.start_land_zone))
        #       free_idcs = np.where(free_space)
        #       self.device_positions = list(zip(free_idcs[1], free_idcs[0]))

        # device들의 좌표 설정
        # self.device_positions = [(x, y) for x in (self.params.x_loc_device) for y in (self.params.y_loc_device)]
        self.device_positions = [(x, y) for x in (self.params.x_loc_device)
                                 for y in (self.params.y_loc_device)]

    def get_comm_obstacles(self):
        return self.map_image.obstacles

    def get_data_map(self):
        return self.device_list.get_data_map(self.shape)

    def get_collected_map(self):
        return self.device_list.get_collected_map(self.shape)

    def get_device_list(self):
        return self.device_list

    def get_grid_params(self):
        return self.params

    def init_episode(self):
        self.device_list = self.device_manager.generate_device_list(self.device_positions)

        if self.params.multi_agent:
            self.num_agents = int(np.random.randint(low=self.params.num_agents_range[0],
                                                    high=self.params.num_agents_range[1] + 1, size=1))
        else:
            self.num_agents = 1
        state = State(self.map_image, self.num_agents, self.params.multi_agent)
        state.reset_devices(self.device_list)

        if self.params.fixed_starting_idcs:
            idx = self.params.starting_idcs
        else:
            # Replace False insures that starting positions of the agents are different
            idx = np.random.choice(len(self.starting_vector), size=self.num_agents, replace=False)
        state.positions = [self.starting_vector[i] for i in idx]

        state.movement_budgets = np.random.randint(low=self.params.movement_range[0],
                                                   high=self.params.movement_range[1] + 1, size=self.num_agents)

        state.initial_movement_budgets = state.movement_budgets.copy()

        return state

    def init_scenario(self, scenario):
        self.device_list = scenario.device_list
        self.num_agents = scenario.init_state.num_agents

        return scenario.init_state

    def get_example_state(self):
        if self.params.multi_agent:
            num_agents = self.params.num_agents_range[0]
        else:
            num_agents = 1
        state = State(self.map_image, num_agents, self.params.multi_agent)
        state.device_map = np.zeros(self.shape, dtype=float)
        state.collected = np.zeros(self.shape, dtype=float)
        return state

#GridActions
from enum import Enum #열거형 enumerate


class GridActions(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    LAND = 4
    HOVER = 5

#GridPhysics
import import_ipynb
from GridActions import GridActions


class GridPhysics:
    def __init__(self):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = None  # state : state.ipynb에서 가져온 듯

    def movement_step(self, action: GridActions):
        old_position = self.state.position  # active_agent의 수 만큼 position 가져 옴
        x, y = old_position

        if action == GridActions.NORTH:
            y += 1
        elif action == GridActions.SOUTH:
            y -= 1
        elif action == GridActions.WEST:
            x -= 1
        elif action == GridActions.EAST:
            x += 1
        elif action == GridActions.LAND:
            self.landing_attempts += 1
            if self.state.is_in_landing_zone():
                # 땅이면 1, 아니면 0 리턴, State.ipynb에서 나옴
                self.state.set_landed(True)  # 이를 통해 해당하는 active_agent는 False에서 True로 바뀐다.
                # 즉, 그 agent는 착륙했다는 것을 의미

        # def set_position(self, position):
        #    self.positions[self.active_agent] = position
        self.state.set_position([x, y])

        # action을 통해 변경된 좌표로 전체 agent의 좌표를 유지하고 있는 self.positions바꿈
        if self.state.is_in_no_fly_zone():
            # Reset state
            self.boundary_counter += 1
            x, y = old_position
            self.state.set_position([x, y])

        self.state.decrement_movement_budget()  # active_agent들을 -1씩 업데이트
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))  # 0 == 0 : True

        # 그 agent가 착륙해있거나 남아있는 movement_budget == 0이 되면 True 리턴
        #       def set_terminal(self, terminal):
        #           self.terminals[self.active_agent] = terminal(self.terminals = [False] in __init__)

        return x, y

    def reset(self, state):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = state

#GridRewards
import import_ipynb
from GridActions import GridActions


class GridRewardParams:
    def __init__(self):
        self.boundary_penalty = 1.0
        self.empty_battery_penalty = 150.0
        self.movement_penalty = 0.2
        self.charging_advantage = 0.02
        self.state = None


class GridRewards:
    def __init__(self, stats):
        self.params = GridRewardParams()
        self.cumulative_reward: float = 0.0

        # ModelStats 분석 후에 다시 하는 걸로
        stats.add_log_data_callback('cumulative_reward', self.get_cumulative_reward)

    #       def add_log_data_callback(self, name: str, callback: callable):
    #           self.log_value_callbacks.append((name, callback))
    #       self.log_value_callbacks = []

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def calculate_motion_rewards(self, state, action: GridActions, next_state):
        reward = 0.0
        if not next_state.landed:  # next_state가 착륙이 아니라면 class state의 landed 함수는 착륙시 True 리턴
            # Penalize battery Consumption
            reward -= self.params.movement_penalty

        # Penalize not moving
        # (This happens when it either 1. tries to land or 2. fly into a boundary or 3. hovers or 4. fly into
        # a cell occupied by another agent)
        # hovering while comm.
        if state.position == next_state.position and not next_state.landed and not action == GridActions.HOVER:
            reward -= self.params.boundary_penalty

        # Penalize battery dead
        if next_state.movement_budget == 0 and not next_state.landed:
            reward -= self.params.empty_battery_penalty

        if self.state.is_in_landing_zone() and not next_state.movement_budget == 0 and next_state.movement_budget < 100:
            reward += self.params.charging_advantage
        # 여기서 'reward = cumulative_reward' 같은 cumulative_reward를 업데이트하는 부분이 나와야 되지 않나
        return reward

    def reset(self):
        self.cumulative_reward = 0

#IoTDevice
import numpy as np
import import_ipynb
from Channel import Channel
#new_data 관련 #### 주석 붙임

class IoTDeviceParams:
    def __init__(self, position=(0, 0), color='blue', data=15.0):
        self.position = position # Device의 좌표들!!!!!
        self.data = data
        self.color = color


class IoTDevice:
    data: float
    collected_data: float
    new_data: float####
    # data_timeseries = []
    # data_rate_timeseries = []

    def __init__(self, params: IoTDeviceParams):
        self.params = params

        self.position = params.position  # fixed position can be later overwritten in reset(Device의 좌표들!!!!!)
        self.color = params.color

        self.data = params.data
        # self.data_timeseries = [self.data]
        # self.data_rate_timeseries = [0]
        self.collected_data = 0
        self.new_data = np.random.uniform(0, 0.5)####

    def collect_data(self, collect):
        if collect == 0:
            return 1
        c = min(collect, self.data - self.collected_data)
        self.collected_data += c

        # return collection ratio, i.e. the percentage of time used for comm
        return c / collect

    @property
    def depleted(self):
        return self.data <= self.collected_data

    def get_data_rate(self, pos, channel: Channel):
        rate = channel.compute_rate(uav_pos=pos, device_pos=self.position)
        # self.data_rate_timeseries.append(rate)
        return rate

    # def log_data(self):
    #     self.data_timeseries.append(self.data - self.collected_data)


class DeviceList: #DeviceList로 디바이스들의 좌표, 데이터 등 정해서 self.devices로 넣는 것 같은데

    def __init__(self, params):
        self.devices = [IoTDevice(device) for device in params]

    def get_data_map(self, shape):
        data_map = np.zeros(shape, dtype=float)

        for device in self.devices:####
            data_map[device.position[1], device.position[0]] = device.data - device.collected_data + device.new_data

        return data_map

    def get_collected_map(self, shape):
        data_map = np.zeros(shape, dtype=float)

        for device in self.devices:
            data_map[device.position[1], device.position[0]] = device.collected_data

        return data_map

    def get_best_data_rate(self, pos, channel: Channel):
        """
        Get the best data rate and the corresponding device index
        (device.depleted 즉, device의 남아있는 data고려해서)
        """
        data_rates = np.array(
            [device.get_data_rate(pos, channel) if not device.depleted else 0 for device in self.devices])
        idx = np.argmax(data_rates) if data_rates.any() else -1
        return data_rates[idx], idx

    def collect_data(self, collect, idx):
        ratio = 1
        if idx != -1: #idx = -1이면 모든 device의 상태가 depleted!!
            ratio = self.devices[idx].collect_data(collect)

        # for device in self.devices:
        #     device.log_data()

        return ratio

    def get_devices(self):
        return self.devices

    def get_device(self, idx):
        return self.devices[idx]

    def get_total_data(self):
        return sum(list([device.data for device in self.devices]))

    def get_collected_data(self):
        return sum(list([device.collected_data for device in self.devices]))

    @property
    def num_devices(self):
        return len(self.devices)

#main
import argparse
import os

import import_ipynb
from MainEnvironment import EnvironmentParams, Environment

from Utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Activates usage of GPU')
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    #parser.add_argument('--config', default=None, help='Path to config file')
    parser.add_argument('--config', default='config/grid_image.json', help='Path to config file')
    parser.add_argument('--id', default='grid_image', help='Overrides the logfile name and the save name')

    #args = parser.parse_args()
    args = parser.parse_args(args=[])

    if args.generate_config:
        generate_config(EnvironmentParams(), "config/default.json")
        exit(0)

    if args.config is None:
        print("Config file needed!")
        exit(1)

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params = read_config(args.config)

    if args.id is not None:
        params.model_stats_params.save_model = "models/" + args.id
        params.model_stats_params.log_file_name = args.id

    env = Environment(params)

    env.run()

#main_mc
#Evaluate a model (saved during training in the 'models' directory)
#through Monte Carlo analysis over the random parameter space for the performance indicators
#'Successful Landing', 'Collection Ratio', 'Collection Ratio and Landed' as defined in the paper
#(plus 'Boundary Counter' counting safety controller activations), e.g. for 1000 Monte Carlo iterations:

import argparse
import os

import numpy as np

import import_ipynb
from MainEnvironment import EnvironmentParams, Environment
from Utils import read_config

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf


def eval_logs(event_path):
    event_acc = EventAccumulator(event_path, size_guidance={'tensors': 100000})
    event_acc.Reload()

    _, _, vals = zip(*event_acc.Tensors('successful_landing'))
    has_landed = [tf.make_ndarray(val) for val in vals]

    _, _, vals = zip(*event_acc.Tensors('cr'))
    cr = [tf.make_ndarray(val) for val in vals]

    _, _, vals = zip(*event_acc.Tensors('cral'))
    cral = [tf.make_ndarray(val) for val in vals]

    _, _, vals = zip(*event_acc.Tensors('boundary_counter'))
    boundary_counter = [tf.make_ndarray(val) for val in vals]

    print("Successful Landing:", sum(has_landed) / len(has_landed))
    print("Collection ratio:", sum(cr) / len(cr))
    print("Collection ratio and landed:", sum(cral) / len(cral))
    print("Boundary counter:", sum(boundary_counter) / len(boundary_counter))


def mc(args, params: EnvironmentParams):
    if args.num_agents is not None:
        num_range = [int(i) for i in args.num_agents]
        params.grid_params.num_agents_range = num_range

    try:
        env = Environment(params)
        env.agent.load_weights(args.weights)

        env.eval(int(args.samples), show=args.show)
    except AttributeError:
        print("Not overriding log dir, eval existing:")

    eval_logs("logs/training/" + args.id + "/test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', required=True, help='Path to weights')
    parser.add_argument('--weights', default='models/grid_image_best', help='Path to weights')
    parser.add_argument('--config', default='config/grid_image.json', help='Config file for agent shaping')
    #parser.add_argument('--config', required=True, help='Config file for agent shaping')
    #parser.add_argument('--id', required=False, help='Id for exported files')
    parser.add_argument('--id', default = 'grid_image', help='Id for exported files')
    parser.add_argument('--samples', default = 1000, help='Id for exported files')
    #parser.add_argument('--samples', required=1000, help='Id for exported files')
    parser.add_argument('--seed', default=None, help="Seed for repeatability")
    parser.add_argument('--show', default=False, help="Show individual plots, allows saving")
    parser.add_argument('--num_agents', default=None, help='Overrides num agents range, argument 12 for range [1,2]')

    #args = parser.parse_args()
    args = parser.parse_args(args=[])

    if args.seed:
        np.random.seed(int(args.seed))

    params = read_config(args.config)

    if args.id is not None:
        params.model_stats_params.save_model = "models/" + args.id
        params.model_stats_params.log_file_name = args.id

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    mc(args, params)

#MainEnvironment
import copy
import distutils.util

import tqdm

import import_ipynb
from Agent import DDQNAgentParams, DDQNAgent
from Trainer import DDQNTrainerParams, DDQNTrainer
from Display import DHDisplay
from Grid import GridParams, Grid
from Physics import PhysicsParams, Physics
from Rewards import RewardParams, Rewards
from State import State
from Environment import BaseEnvironment, BaseEnvironmentParams
from GridActions import GridActions


class EnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = GridParams()
        self.reward_params = RewardParams()
        self.trainer_params = DDQNTrainerParams()
        self.agent_params = DDQNAgentParams()
        self.physics_params = PhysicsParams()


class Environment(BaseEnvironment):
    def __init__(self, params: EnvironmentParams):
        self.display = DHDisplay()
        super().__init__(params, self.display)

        self.grid = Grid(params.grid_params, stats=self.stats)
        self.rewards = Rewards(params.reward_params, stats=self.stats)
        self.physics = Physics(params=params.physics_params, stats=self.stats)
        self.agent = DDQNAgent(params.agent_params, self.grid.get_example_state(),
                               self.physics.get_example_action(), stats=self.stats)
        self.trainer = DDQNTrainer(params.trainer_params, agent=self.agent)

        self.display.set_channel(self.physics.channel)

        self.first_action = True
        self.last_actions = []
        self.last_rewards = []
        self.last_states = []

    def test_episode(self):
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        first_action = True
        while not state.all_terminal:
            for state.active_agent in range(state.num_agents):
                if state.terminal:
                    continue # continue 실행될 경우 아래 코드 실행되지 않고 건너뛴 뒤 다음 반복 시작
                action = self.agent.get_exploitation_action_target(state)
                if not first_action:
                    reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                           GridActions(self.last_actions[state.active_agent]),
                                                           state)
                    self.stats.add_experience(
                        (self.last_states[state.active_agent], self.last_actions[state.active_agent],
                         reward, copy.deepcopy(state)))

                self.last_states[state.active_agent] = copy.deepcopy(state) #state 이동
                self.last_actions[state.active_agent] = action #action 이동
                state = self.physics.step(GridActions(action))
                if state.terminal:
                    reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                           GridActions(self.last_actions[state.active_agent]),
                                                           state)
                    self.stats.add_experience(
                        (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward,
                         copy.deepcopy(state)))

            first_action = False

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_testing_data(step=self.step_count)

    def test_scenario(self, scenario):
        state = copy.deepcopy(self.init_episode(scenario))
        while not state.all_terminal:
            for state.active_agent in range(state.num_agents):
                if state.terminal:
                    continue
                action = self.agent.get_exploitation_action_target(state)
                state = self.physics.step(GridActions(action))

    def step(self, state: State, random=False):
        for state.active_agent in range(state.num_agents):
            if state.terminal:
                continue
            if random:
                action = self.agent.get_random_action()
            else:
                action = self.agent.act(state)
            if not self.first_action:
                reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                       GridActions(self.last_actions[state.active_agent]), state)
                self.trainer.add_experience(self.last_states[state.active_agent], self.last_actions[state.active_agent],
                                            reward, state)
                self.stats.add_experience(
                    (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward,
                     copy.deepcopy(state)))

            self.last_states[state.active_agent] = copy.deepcopy(state)
            self.last_actions[state.active_agent] = action
            state = self.physics.step(GridActions(action))
            if state.terminal:
                reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                       GridActions(self.last_actions[state.active_agent]), state)
                self.trainer.add_experience(self.last_states[state.active_agent], self.last_actions[state.active_agent],
                                            reward, state)
                self.stats.add_experience(
                    (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward,
                     copy.deepcopy(state)))

        self.step_count += 1
        self.first_action = False
        return state

    def init_episode(self, init_state=None):
        state = super().init_episode(init_state)
        #BaseEnvironment 클래스의 init_episode
        self.last_states = [None] * state.num_agents
        self.last_actions = [None] * state.num_agents
        self.first_action = True
        return state

#Map
import numpy as np
% matplotlib
inline
import matplotlib.pyplot as plt
from skimage import io


#
# land size : 125 - 134, 62 - 77
class Map:
    # def __init__(self, map_data):
    def __init__(self, map_data):
        self.start_land_zone = map_data[:, :, 2].astype(bool)
        self.nfz = map_data[:, :, 0].astype(bool)
        self.obstacles = map_data[:, :, 1].astype(bool)

    def get_starting_vector(self):
        similar = np.where(self.start_land_zone)
        return list(zip(similar[1], similar[0]))

    def get_free_space_vector(self):
        free_space = np.logical_not(np.logical_or(self.obstacles, self.start_land_zone))
        free_idcs = np.where(free_space)
        return list(zip(free_idcs[1], free_idcs[0]))

    def get_size(self):
        return self.start_land_zone.shape[:2]


def load_image(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return np.array(data, dtype=bool)


def save_image(path, image):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    if image.dtype == bool:
        io.imsave(path, image * np.uint8(255))
    else:
        io.imsave(path, image)


def load_map(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=False)
    return Map(data)


def load_target(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return np.array(data, dtype=bool)

#ModelStats
import import_ipynb
import collections
import datetime
import os
import shutil  # shutil 모듈은 파일과 파일 모음에 대한 여러 가지 고수준 연산을 제공합니다. 특히, 파일 복사와 삭제를 지원하는 함수가 제공됩니다.

import tensorflow as tf
import numpy as np
import distutils.util


class ModelStatsParams:
    def __init__(self,
                 save_model='models/save_model',
                 moving_average_length=50):
        self.save_model = save_model
        self.moving_average_length = moving_average_length
        self.log_file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # '20210511-211656' : 년/달/날짜 - 시/분/초
        self.training_images = False


class ModelStats:

    def __init__(self, params: ModelStatsParams, display, force_override=False):
        self.params = params
        self.display = display  # ?

        self.evaluation_value_callback = None
        self.env_map_callback = None
        self.log_value_callbacks = []
        self.trajectory = []

        self.log_dir = "logs/training/" + params.log_file_name
        # ex) 'logs/training/20210511-212027'
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                                   histogram_freq=100)
        # log_dir : the path of the directory where to save the log files to be parsed by TensorBoard.
        # histogram_freq : frequency (in epochs) at which to compute activation and weight histograms for the layers of the model.
        self.model = None

        if os.path.isdir(self.log_dir):  # self.log_dir이 존재할 경우 True 리턴
            if force_override:
                shutil.rmtree(self.log_dir)  # 지정된 폴더와 하위 디렉토리 폴더, 파일를 모두 삭제
            else:
                print(self.log_dir, 'already exists.')
                resp = input('Override log file? [Y/n]\n')  # Y or n을 input으로 받음
                if resp == '' or distutils.util.strtobool(resp):
                    # rest == '' : input에서 아무것도 치지 않음(즉, 빈칸 입력)
                    # strtobool : Convert a string representation of truth to true (1) or false (0).
                    # strtobool('true', '1', 'y', 'yes') = 1
                    print('Deleting old log dir')
                    shutil.rmtree(self.log_dir)  # 지정된 폴더와 하위 디렉토리 폴더, 파일를 모두 삭제
                else:
                    raise AttributeError('Okay bye')  # raise : 사용자가 직접 오류를 일으킴

        self.training_log_writer = tf.summary.create_file_writer(self.log_dir + '/training')
        self.testing_log_writer = tf.summary.create_file_writer(self.log_dir + '/test')
        # tf.summary.create_file_writer : 주어진 log_dir로 summary file 만듬

        self.evaluation_deque = collections.deque(maxlen=params.moving_average_length)
        # maxlen=params.moving_average_length인 비어있는 deque(double-ended-queue) 생성
        self.eval_best = -float('inf')  # -float('inf') = -inf
        self.bar = None

    def set_evaluation_value_callback(self, callback: callable):
        # callback에 들어오는 것을 callable 즉, 객체로호출가능하게 설정하고 그 객체를 def에서 사용(이렇게 이해함)
        # physics.ipynb에서 callback = get_cral로 사용함
        # get_cral = self.get_collection_ratio() * self.state.all_landed
        self.evaluation_value_callback = callback

    # environoment.ipynb에서 사용함
    def add_experience(self, experience):
        self.trajectory.append(experience)

    # Agent.ipynb에서 사용함
    def set_model(self, model):
        self.tensorboard_callback.set_model(model)
        # set_model : Sets Keras model and writes graph if specified.
        self.model = model

    # BaseGrid.ipynb에서 사용함
    def set_env_map_callback(self, callback: callable):
        self.env_map_callback = callback

    # GridRewards.ipynb에서 사용
    def add_log_data_callback(self, name: str, callback: callable):
        self.log_value_callbacks.append((name, callback))

    def log_training_data(self, step):

        with self.training_log_writer.as_default():
            # with구문에서 self.training_log_writer을 기본값으로 설정
            # with구문은 코드 실행이 시작 할 때 설정이 필요하고 코드가 종료 되는
            # 시점에 해제가 필요한 경우에 사용하면 편리한 문법이다.
            self.log_data(step, self.params.training_images)

    def log_testing_data(self, step):
        with self.testing_log_writer.as_default():
            # with구문에서 self.testig_log_writer을 기본값으로 설정
            self.log_data(step)

        if self.evaluation_value_callback:
            # get_collection_ratio()의 값은 모르지만 state.all_landed 즉, 모든 agent가 land -> True가 되면
            # evaluationo_deque에 추가를 한다. get_collection_ratio 값을
            self.evaluation_deque.append(self.evaluation_value_callback())

    def log_data(self, step, images=True):

        for callback in self.log_value_callbacks:
            tf.summary.scalar(callback[0], callback[1](), step=step)
            # scalar summary 작성(주로 accuracy, cost(loss)와 같은 scalar 텐서에 사용)
            # callback[1]()에서 ()있는 이유는 GridRewards에서 def get_cumulative_reward(self):로 정의를 했기 때문
        if images:
            trajectory = self.display.display_episode(self.env_map_callback(), trajectory=self.trajectory)
            tf.summary.image('trajectory', trajectory,
                             step=step)
            # image가 있다면 image형 summary 제시

    def save_if_best(self):
        if len(self.evaluation_deque) < self.params.moving_average_length:
            # self.evaluation_deque의 길이가 moving_average_length임을 감안하면 deque = 50으로 가득찼을 때
            # 아래 함수 실행
            return

        eval_mean = np.mean(self.evaluation_deque)  # deque에 대해서 전체 평균 내기
        if eval_mean > self.eval_best:
            self.eval_best = eval_mean
            if self.params.save_model != '':
                print('Saving best with:', eval_mean)
                self.model.save_weights(self.params.save_model + '_best')
                # 해당 경로의 model의 weights 저장 'models/save_model' + '_best'

    def get_log_dir(self):
        return self.log_dir

    def training_ended(self):

        if self.params.save_model != '':
            self.model.save_weights(self.params.save_model + '_unfinished')
            print('Model saved as', self.params.save_model + '_unfinished')

    def save_episode(self, save_path):
        f = open(save_path + ".txt", "w")
        # 파일이름 , w : 쓰기모드

        for callback in self.log_value_callbacks:
            f.write(callback[0] + ' ' + str(callback[1]()) + '\n')
        f.close()

    def on_episode_begin(self, episode_count):
        self.tensorboard_callback.on_epoch_begin(episode_count)  # Called at the start of an epoch.(only train)
        self.trajectory = []

    def on_episode_end(self, episode_count):
        self.tensorboard_callback.on_epoch_end(episode_count)  # Called at the end of an epoch.(only train)

#Physics
import numpy as np

import import_ipynb
from Channel import ChannelParams, Channel
from State import State
from ModelStats import ModelStats
from GridActions import GridActions
from GridPhysics import GridPhysics


class PhysicsParams:
    def __init__(self):
        self.channel_params = ChannelParams()
        self.comm_steps = 4


class Physics(GridPhysics):

    def __init__(self, params: PhysicsParams, stats: ModelStats):

        super().__init__()

        self.channel = Channel(params.channel_params)

        self.params = params

        self.register_functions(stats)

    def register_functions(self, stats: ModelStats):
        stats.set_evaluation_value_callback(self.get_cral)

        stats.add_log_data_callback('cral', self.get_cral)
        stats.add_log_data_callback('cr', self.get_collection_ratio)
        stats.add_log_data_callback('successful_landing', self.has_landed)
        stats.add_log_data_callback('boundary_counter', self.get_boundary_counter)
        stats.add_log_data_callback('landing_attempts', self.get_landing_attempts)
        stats.add_log_data_callback('movement_ratio', self.get_movement_ratio)

    def reset(self, state: State):
        GridPhysics.reset(self, state)

        self.channel.reset(self.state.shape[0])

    def step(self, action: GridActions):
        old_position = self.state.position
        self.movement_step(action)
        if not self.state.terminal:
            self.comm_step(old_position)

        return self.state

    def comm_step(self, old_position):
        positions = list(
            reversed(np.linspace(self.state.position, old_position, num=self.params.comm_steps, endpoint=False)))
        # reversed : list 객체의 순서를 반대로
        # np.linspace(start, stop, num) : 시작점부터 끝점까지 균일하게 잘라줌

        indices = []
        device_list = self.state.device_list
        for position in positions:
            data_rate, idx = device_list.get_best_data_rate(position, self.channel)
            # (device.depleted(모아진 데이터가 더 많으면 depleted라고 판단)
            # device마다의 data rate를 구해서 그 중에서 가장 큰 노드와 통신
            device_list.collect_data(data_rate, idx)
            # collect ratio 리턴
            indices.append(idx)
            # 통신할 노드 index 리턴

        self.state.collected = device_list.get_collected_map(self.state.shape)
        # 각 노드별 모아온 data
        self.state.device_map = device_list.get_data_map(self.state.shape)
        # 각 노드별 통신을 하고 남아있는 data

        idx = max(set(indices), key=indices.count)
        # set : 중복되지 않은 원소를 얻고자 할 때, dictionoary 형태
        # max(keyi = ) : indices 중 가장 많이 중복된 요소를 선택해라
        self.state.set_device_com(idx)
        # 활성화된 agent는 어떤 index를 가진 노드와 통신을 할 것이다.

        return idx

    def get_example_action(self):
        return GridActions.HOVER

    def is_in_landing_zone(self):
        return self.state.is_in_landing_zone()

    def get_collection_ratio(self):
        return self.state.get_collection_ratio()

    def get_movement_budget_used(self):
        return sum(self.state.initial_movement_budgets) - sum(self.state.movement_budgets)

    def get_max_rate(self):
        return self.channel.get_max_rate()

    def get_average_data_rate(self):
        return self.state.get_collected_data() / self.get_movement_budget_used()

    def get_cral(self):
        return self.get_collection_ratio() * self.state.all_landed

    def get_boundary_counter(self):
        return self.boundary_counter

    def get_landing_attempts(self):
        return self.landing_attempts

    def get_movement_ratio(self):
        return float(self.get_movement_budget_used()) / float(sum(self.state.initial_movement_budgets))

    def has_landed(self):
        return self.state.all_landed

#ReplayMemory
import numpy as np


def shape(exp):
    if type(exp) is np.ndarray:
        return list(exp.shape)
    else:
        return []


def type_of(exp):
    if type(exp) is np.ndarray:
        return exp.dtype  # exp의 데이터 타입 리턴
    else:
        return type(exp)  # a의 타입 확인


class ReplayMemory(object):
    """
    Replay memory class for RL
    """

    def __init__(self, size):
        self.k = 0
        self.head = -1
        self.full = False
        self.size = size
        self.memory = None

    def initialize(self, experience):
        self.memory = [np.zeros(shape=[self.size] + shape(exp), dtype=type_of(exp)) for exp in experience]
        # np.zeros(shape = [2,2] + [1]), np.zeros(shape = [2,2] + [2]) 등의 순서로 for 구문만큼 생성

    def store(self, experience):
        if self.memory is None:
            self.initialize(experience)
        if len(experience) != len(self.memory):
            raise Exception('Experience not the same size as memory', len(experience), '!=', len(self.memory))

        for e, mem in zip(experience, self.memory):
            mem[self.k] = e  # memory에 exp를 넣어라는 뜻

        self.head = self.k
        self.k += 1
        if self.k >= self.size:
            self.k = 0
            self.full = True

    def sample(self, batch_size):
        r = self.size
        if not self.full:
            r = self.k
        random_idx = np.random.choice(r, size=batch_size, replace=False)  # r 중 batch size만큼 샘플링 0 5 3
        random_idx[0] = self.head  # Always add the latest one # -1  5 3

        return [mem[random_idx] for mem in self.memory]

    def get(self, start, length):
        return [mem[start:start + length] for mem in self.memory]

    def get_size(self):
        if self.full:
            return self.size
        return self.k

    def get_max_size(self):
        return self.size

    def reset(self):
        self.k = 0
        self.head = -1
        self.full = False

    def shuffle(self):
        self.memory = self.sample(self.get_size())

#Rewards
import import_ipynb
from State import State
from GridActions import GridActions
from GridRewards import GridRewards, GridRewardParams


class RewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()
        self.data_multiplier = 1.0


# Class used to track rewards
class Rewards(GridRewards):
    cumulative_reward: float = 0.0

    def __init__(self, reward_params: RewardParams, stats):
        super().__init__(stats)
        self.params = reward_params
        self.reset()

    def calculate_reward(self, state: State, action: GridActions, next_state: State):
        reward = self.calculate_motion_rewards(state, action, next_state)

        # Reward the collected data
        reward += self.params.data_multiplier * (state.get_remaining_data() - next_state.get_remaining_data())

        # Cumulative reward
        self.cumulative_reward += reward

        return reward

#Shadowing
import numpy as np
import os
import tqdm
import import_ipynb
from Map import load_map


def bresenham(x0, y0, x1, y1, obstacles, shadow_map):
    # 선그리기 알고리즘
    if obstacles[y0, x0]:
        return
    x_dist = abs(x0 - x1)
    y_dist = -abs(y0 - y1)
    x_step = 1 if x1 > x0 else -1
    y_step = 1 if y1 > y0 else -1

    error = x_dist + y_dist

    # shadowed = False
    shadow_map[y0, x0] = False

    while x0 != x1 or y0 != y1:
        # 둘 중 하나가 같아질 때 까지
        if 2 * error - y_dist > x_dist - 2 * error:
            # x축으로 더 멀다면(즉, abs(x_dist) > abs(y_dist)
            # horizontal step
            error += y_dist
            x0 += x_step
        else:
            # vertical step
            error += x_dist
            y0 += y_step

        if obstacles[y0, x0]:
            # shadowed = True
            return

        # if shadowed:
        shadow_map[y0, x0] = False


def calculate_shadowing(map_path, save_as):  # 전체 맵에서 어디 좌표가 shadowing이 되는지를 True로 리턴

    total_map = load_map(map_path)
    obstacles = total_map.obstacles  # 하지만 나의 환경에는 장애물이 없다.
    size = total_map.obstacles.shape[0]  # size = 140
    total = size * size

    total_shadow_map = np.ones((size, size, size, size), dtype=bool)
    with tqdm.tqdm(total=total) as pbar:
        for i, j in np.ndindex(total_map.obstacles.shape):
            # i, j : (0,0), (0,1),,,(0,139),(1,0), ,,, (139, 139) 즉, 140 x 140의 모든 좌표 반복
            shadow_map = np.ones((size, size), dtype=bool)

            for x in range(size):
                bresenham(i, j, x, 0, obstacles, shadow_map)
                # bresenham : 두 좌표 사이에 shadowing이 있는지를 판단하는거고 있으면 shadow_map에 True로 반환
                bresenham(i, j, x, size - 1, obstacles, shadow_map)
                bresenham(i, j, 0, x, obstacles, shadow_map)
                bresenham(i, j, size - 1, x, obstacles, shadow_map)

            total_shadow_map[j, i] = shadow_map
            pbar.update(1)

    np.save(save_as, total_shadow_map)
    return total_shadow_map


def load_or_create_shadowing(map_path):
    shadow_file_name = os.path.splitext(map_path)[0] + "_shadowing.npy"
    # splitext : 확장자만 따로 떨어트림
    if os.path.exists(shadow_file_name):
        return np.load(shadow_file_name)
    else:
        return calculate_shadowing(map_path, shadow_file_name)

#State
import numpy as np
import import_ipynb
from Map import Map
from StateUtils import pad_centered
from BaseState import BaseState


class State(BaseState):  # property : __init__의 속성값 가져오기
    def __init__(self, map_init: Map, num_agents: int, multi_agent: bool):
        super().__init__(map_init)
        self.device_list = None
        self.device_map = None  # Floating point sparse matrix showing devices and their data to be collected

        # Multi-agent active agent decides on properties
        self.active_agent = 0
        self.num_agents = num_agents
        self.multi_agent = multi_agent

        # Multi-agent is creating lists
        self.positions = [[0, 0]] * num_agents
        self.movement_budgets = [0] * num_agents
        self.landeds = [False] * num_agents
        self.terminals = [False] * num_agents
        self.device_coms = [-1] * num_agents

        self.initial_movement_budgets = [0] * num_agents
        self.initial_total_data = 0
        self.collected = None

    @property
    def position(self):
        return self.positions[self.active_agent]

    @property
    def movement_budget(self):
        return self.movement_budgets[self.active_agent]

    @property
    def initial_movement_budget(self):
        return self.initial_movement_budgets[self.active_agent]

    @property
    def landed(self):
        return self.landeds[self.active_agent]

    @property
    def terminal(self):
        return self.terminals[self.active_agent]

    @property
    def all_landed(self):
        return all(self.landeds)  # 파이썬 내장 함수로 해당 요소가 모두 참이면 True, 단 하나라도 아니면 False 리턴

    @property
    def all_terminal(self):
        return all(self.terminals)

    def is_terminal(self):
        return self.all_terminal

    # GridPhysics에서 사용되는데 is_in_land(땅이면 1, 아니면 0 리턴)을 통해 땅이면 True를집어넣게끔
    def set_landed(self, landed):
        self.landeds[self.active_agent] = landed

    def set_position(self, position):
        self.positions[self.active_agent] = position

    def decrement_movement_budget(self):
        self.movement_budgets[self.active_agent] -= 1

    def increment_movement_budget(self):
        self.movement_budgets[self.active_agent] += 5

    def set_terminal(self, terminal):
        self.terminals[self.active_agent] = terminal

    def set_device_com(self, device_com):
        self.device_coms[self.active_agent] = device_com

    def get_active_agent(self):
        return self.active_agent

    def get_remaining_data(self):
        return np.sum(self.device_map)

    def get_total_data(self):
        return self.initial_total_data

    def get_scalars(self, give_position=False):  # 각자 독립적으로 대하기 때문에 좌표없이 스칼라값을 리턴한다.
        if give_position:  # 좌표를 안주기 때문에 if구문 실행안한다고 보면 된다.
            return np.array([self.movement_budget, self.position[0], self.position[1]])

        return np.array([self.movement_budget])  # 활성화중인 agent의 남은 budget이 나옴

    def get_num_scalars(self, give_position=False):
        return len(self.get_scalars(give_position))

    def get_boolean_map(self):
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)
        if self.multi_agent:
            padded_rest = pad_centered(self,
                                       np.concatenate(
                                           [np.expand_dims(self.landing_zone, -1), self.get_agent_bool_maps()],
                                           axis=-1), 0)
        else:
            padded_rest = pad_centered(self, np.expand_dims(self.landing_zone, -1), 0)

        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    def get_float_map(self):
        if self.multi_agent:
            return pad_centered(self, np.concatenate([np.expand_dims(self.device_map, -1),
                                                      self.get_agent_float_maps()], axis=-1), 0)
        else:
            return pad_centered(self, np.expand_dims(self.device_map, -1), 0)

    def get_float_map_shape(self):
        return self.get_float_map().shape

    def is_in_landing_zone(self):
        return self.landing_zone[self.position[1]][self.position[0]]

    def is_in_no_fly_zone(self):
        # Out of bounds is implicitly nfz
        if 0 <= self.position[1] < self.no_fly_zone.shape[0] and 0 <= self.position[0] < self.no_fly_zone.shape[1]:
            # NFZ or occupied
            return self.no_fly_zone[self.position[1], self.position[0]] or self.is_occupied()
        return True

    def is_occupied(self):
        # 아직 i index의 UAV가 터미널 도착하지도 않았고 active_agent도 아닐 때 active_agent의 좌표와 일치시 True
        if not self.multi_agent:
            return False
        for i, pos in enumerate(self.positions):  # i : index of UAV, pos : 그 UAV의 좌표
            if self.terminals[i]:
                continue  # 아래 문장을 실행하지 않고 다음 반복(위의 for구문)을 시작
            if i == self.active_agent:
                continue  # 아래 문장을 실행하지 않고 다음 반복(위의 for구문)을 시작
            if pos == self.position:
                return True
        return False

    def get_collection_ratio(self):
        return np.sum(self.collected) / self.initial_total_data

    def get_collected_data(self):
        return np.sum(self.collected)

    ######################################################################################
    ##중요해보이는 함수
    # device_list : Grid.ipynb에서 정의함
    def reset_devices(self, device_list):
        # get_data_map(IoTDevice에 나옴) : 각 device들의 좌표에 data 초기값(15) - collected_data한 data_map 리턴
        self.device_map = device_list.get_data_map(self.no_fly_zone.shape)
        self.collected = np.zeros(self.no_fly_zone.shape, dtype=float)
        self.initial_total_data = device_list.get_total_data()
        self.device_list = device_list

    ######################################################################################

    def get_agent_bool_maps(self):
        agent_map = np.zeros(self.no_fly_zone.shape + (1,), dtype=bool)
        # (140, 140) -> (140, 140, 1) filled with 'False'
        for agent in range(self.num_agents):
            # self.positions = [[0, 0]] * num_agents
            # agent_map[self.positions[agent][1], self.positions[agent][0]][0] = self.landeds[agent]
            agent_map[self.positions[agent][1], self.positions[agent][0]][0] = not self.terminals[agent]
            # agent_map = (140,140,1)의 False 행렬인데 각 agent가 위치한 좌표에는
            # terminal에 도달하지 않은 agent들을 not False(True)로 반환
            # (terminal에 도달했으면 true -> not true = False니까 관계없음)
        return agent_map

    def get_agent_float_maps(self):
        agent_map = np.zeros(self.no_fly_zone.shape + (1,), dtype=float)
        # (140, 140) -> (140, 140, 1) filled with '0.'
        for agent in range(self.num_agents):
            agent_map[self.positions[agent][1], self.positions[agent][0]][0] = self.movement_budgets[agent]
            # 현재 맵에서 agent들이 있는 위치에 그 agent에 남아있는 movement_budgets을 나타냄
        return agent_map

    # max_num_devices, ralative : agent.ipynb에 나옴(max_num_devices = 10으로 나옴, relative : False로 나옴)
    def get_device_scalars(self, max_num_devices, relative):
        devices = np.zeros(3 * max_num_devices, dtype=np.float32)
        # devices : (3 * max_num_devices, 1)의 형태
        if relative:
            for k, dev in enumerate(self.device_list.devices):
                # for in enumerate : index, value 리턴
                # .devices가 정확히 의미하는 바는 찾지 못함
                devices[k * 3] = dev.position[0] - self.position[0]
                devices[k * 3 + 1] = dev.position[1] - self.position[1]
                devices[k * 3 + 2] = dev.data - dev.collected_data  # 그 device의 남아있는 data를 의미함
        else:
            for k, dev in enumerate(self.device_list.devices):
                devices[k * 3] = dev.position[0]
                devices[k * 3 + 1] = dev.position[1]
                devices[k * 3 + 2] = dev.data - dev.collected_data  # 그 device의 남아있는 data를 의미함
        return devices

    # agnet.py서 활용됨(max_num_uavs = 3으로,,)
    def get_uav_scalars(self, max_num_uavs, relative):
        uavs = np.zeros(4 * max_num_uavs, dtype=np.float32)
        if relative:
            for k in range(max_num_uavs):
                if k >= self.num_agents:
                    break  # 반복문(여기서는 for구문) 끝내기
                uavs[k * 4] = self.positions[k][0] - self.position[0]
                uavs[k * 4 + 1] = self.positions[k][1] - self.position[1]
                uavs[k * 4 + 2] = self.movement_budgets[k]
                uavs[k * 4 + 3] = not self.terminals[k]
        else:
            for k in range(max_num_uavs):
                if k >= self.num_agents:
                    break
                uavs[k * 4] = self.positions[k][0]
                uavs[k * 4 + 1] = self.positions[k][1]
                uavs[k * 4 + 2] = self.movement_budgets[k]
                uavs[k * 4 + 3] = not self.terminals[k]
        return uavs

#StateUtils
import math
import numpy as np

def pad_centered(state, map_in, pad_value):
    padding_rows = math.ceil(state.no_fly_zone.shape[0] / 2.0) #올림 70
    padding_cols = math.ceil(state.no_fly_zone.shape[1] / 2.0) #올림 70
    position_x, position_y = state.position
    position_row_offset = padding_rows - position_y
    position_col_offset = padding_cols - position_x
    return np.pad(map_in,
                  pad_width=[[int(abs(padding_rows + position_row_offset - 1) / 4), #위쪽
                              int(abs(padding_rows - position_row_offset)/ 4)],  #아래쪽
                             [int(abs(padding_cols + position_col_offset - 1)/ 4), #왼쪽
                              int(abs(padding_cols - position_col_offset)/ 4)], #오른쪽
                             [0, 0]], #z축에 대한 pad 변경
                  mode='constant', #항상 똑같은 상수값 채워넣기
                  constant_values=pad_value)

#Trainer
import import_ipynb
from Agent import DDQNAgent
from ReplayMemory import ReplayMemory
import tqdm


class DDQNTrainerParams:
    def __init__(self):
        self.batch_size = 128
        #self.batch_size = 32
        #self.num_steps = 1e6
        self.num_steps = 5e4
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 5
        self.rm_size = 50000
        self.load_model = ""


class DDQNTrainer:
    def __init__(self, params: DDQNTrainerParams, agent: DDQNAgent):
        self.params = params
        self.replay_memory = ReplayMemory(size=params.rm_size)
        self.agent = agent
        self.use_scalar_input = self.agent.params.use_scalar_input

        if self.params.load_model != "":
            print("Loading model", self.params.load_model, "for agent")
            self.agent.load_weights(self.params.load_model)

        self.prefill_bar = None

    def add_experience(self, state, action, reward, next_state):
        if self.use_scalar_input:
            self.replay_memory.store((state.get_device_scalars(self.agent.params.max_devices, self.agent.params.relative_scalars),
                                      state.get_uav_scalars(self.agent.params.max_uavs, self.agent.params.relative_scalars),
                                      state.get_scalars(give_position=True),
                                      action,
                                      reward,
                                      next_state.get_device_scalars(self.agent.params.max_devices, self.agent.params.relative_scalars),
                                      next_state.get_uav_scalars(self.agent.params.max_uavs, self.agent.params.relative_scalars),
                                      next_state.get_scalars(give_position=True),
                                      next_state.terminal))
        else:
            self.replay_memory.store((state.get_boolean_map(),
                                      state.get_float_map(),
                                      state.get_scalars(),
                                      action,
                                      reward,
                                      next_state.get_boolean_map(),
                                      next_state.get_float_map(),
                                      next_state.get_scalars(),
                                      next_state.terminal))

    def train_agent(self):
        if self.params.batch_size > self.replay_memory.get_size():
            return
        mini_batch = self.replay_memory.sample(self.params.batch_size)

        self.agent.train(mini_batch)

    def should_fill_replay_memory(self):
        target_size = self.replay_memory.get_max_size() * self.params.rm_pre_fill_ratio
        if self.replay_memory.get_size() >= target_size or self.replay_memory.full:
            if self.prefill_bar:
                self.prefill_bar.close()
            return False

        if self.prefill_bar is None:
            print("Filling replay memory")
            self.prefill_bar = tqdm.tqdm(total=target_size)

        self.prefill_bar.update(self.replay_memory.get_size() - self.prefill_bar.n)

        return True

#Utils
import distutils
import json

from types import SimpleNamespace as Namespace


def getattr_recursive(obj, s):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    return getattr_recursive(getattr(obj, split[0]), split[1:]) if len(split) > 1 else getattr(obj, split[0])


def setattr_recursive(obj, s, val):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')
    return setattr_recursive(getattr(obj, split[0]), split[1:], val) if len(split) > 1 else setattr(obj, split[0], val)


def generate_config(params, file_path):
    print("Saving Configs")
    f = open(file_path, "w")
    json_data = json.dumps(params.__dict__, default=lambda o: o.__dict__, indent=4)
    f.write(json_data)
    f.close()


def read_config(config_path):
    print('Parse Params file here from ', config_path, ' and pass into main')
    json_data = open(config_path, "r").read()
    return json.loads(json_data, object_hook=lambda d: Namespace(**d))