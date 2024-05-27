import time
import matplotlib.pyplot as plt
import numpy as np
import math
import random

# 定义节点类
class Node:
    def __init__(self, x, y):
        self.x = x  # x坐标
        self.y = y  # y坐标
        self.parent = None  # 父节点
        self.cost = 0.0  # 当前路径开销

# 定义 RRT 算法类
class myRRT:
    def __init__(self, start, goal, map_dims, obstacle_list, expand_dis=1.0, min_expand_dis=1.0,max_expand_dis=3.0,goal_sample_rate=0.1, max_iter=500):
        self.start = Node(start[0], start[1])   # 起始节点
        self.goal = Node(goal[0], goal[1])  # 目标节点
        self.map_width, self.map_height = map_dims  # 地图宽高
        self.obstacle_list = obstacle_list  # 障碍物列表
        self.expand_dis = expand_dis  # 步长（rrt_star, informed_rrt使用）
        self.min_expand_dis = min_expand_dis  # 最小步长（dynamic_rrt使用）
        self.max_expand_dis = max_expand_dis  # 最大步长（dynamic_rrt使用）
        self.goal_sample_rate = goal_sample_rate  # 采样目标点的概率
        self.max_iter = max_iter  # 最大迭代次数
        self.node_list = [self.start]  # 初始化节点列表

    # rrt_star规划路径
    def rrt_star_planning(self):
        for i in range(self.max_iter):
            rand_node = self.get_random_node()  # 随机采样节点
            nearest_idx = self.get_nearest_node_index(self.node_list, rand_node)  # 找到最近的节点索引
            nearest_node = self.node_list[nearest_idx]
            new_node,theta,d = self.steer(nearest_node, rand_node, self.expand_dis)  # 扩展新节点
            if self.check_collision_extend(nearest_node, theta, d):  # 检查路径是否与障碍物碰撞
                near_inds = self.find_near_nodes(new_node)  # 找到附近的节点
                new_node = self.choose_parent(new_node, near_inds)  # 选择新节点的父节点
                self.node_list.append(new_node)  # 将新节点添加到节点列表
                self.rewire(new_node, near_inds)  # 重新连接节点
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node,theta,d = self.steer(self.node_list[-1], self.goal, self.expand_dis)
                if self.check_collision(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)
        return None
    
    # informed_rrt规划路径
    def informed_rrt_planning(self):
        best_cost = float("inf")
        best_path = None
        for i in range(self.max_iter):
            rand_node = self.get_informed_random_node(best_cost)    # 启发式采样
            nearest_idx = self.get_nearest_node_index(self.node_list, rand_node)  # 找到最近的节点索引
            nearest_node = self.node_list[nearest_idx]
            new_node,theta,d = self.steer(nearest_node, rand_node, self.expand_dis)  # 扩展新节点
            if self.check_collision_extend(nearest_node, theta, d):  # 检查路径是否与障碍物碰撞
                self.node_list.append(new_node)  # 将新节点添加到节点列表
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node,theta,d = self.steer(self.node_list[-1], self.goal, self.expand_dis)
                if self.check_collision(final_node):
                    new_path = self.generate_final_course(len(self.node_list) - 1)
                    new_cost = self.calc_cost(new_path)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_path = new_path
        return best_path

    # dynamic_rrt规划路径
    def dynamic_rrt_planning(self):
        for i in range(self.max_iter):
            rand_node = self.get_random_node()  # 随机采样节点
            nearest_idx = self.get_nearest_node_index(self.node_list, rand_node)  # 找到最近的节点索引
            nearest_node = self.node_list[nearest_idx]
            dynamic_expand_dis = self.dynamic_expand_dis(nearest_node)  # 动态步长
            new_node,theta,d = self.steer(nearest_node, rand_node, dynamic_expand_dis)  # 扩展新节点
            if self.check_collision_extend(nearest_node, theta, d):  # 检查路径是否与障碍物碰撞
                self.node_list.append(new_node)  # 将新节点添加到节点列表
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.min_expand_dis:
                final_node,theta,d = self.steer(self.node_list[-1], self.goal, self.min_expand_dis)
                if self.check_collision(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)
        return None
    
    # 计算当前动态步长
    def dynamic_expand_dis(self, node):
        min_dist_to_obstacle = float("inf")
        for ox, oy, size in self.obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            dist = math.sqrt(dx ** 2 + dy ** 2) - size
            if dist < min_dist_to_obstacle:
                min_dist_to_obstacle = dist
        # 确保步长在最小步长和最大步长之间
        dynamic_dis = max(self.min_expand_dis, min(self.max_expand_dis, min_dist_to_obstacle / 2))
        return dynamic_dis

    # 选择新节点的父节点
    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return new_node
        dlist = []
        for i in near_inds:
            dx = new_node.x - self.node_list[i].x
            dy = new_node.y - self.node_list[i].y
            d = math.sqrt(dx ** 2 + dy ** 2)
            theta = math.atan2(dy, dx)
            if self.check_collision_extend(self.node_list[i], theta, d):
                dlist.append(self.node_list[i].cost + d)
            else:
                dlist.append(float("inf"))
        min_cost = min(dlist)
        min_ind = near_inds[dlist.index(min_cost)]
        if min_cost == float("inf"):
            return new_node
        new_node.cost = min_cost
        new_node.parent = self.node_list[min_ind]
        return new_node

    # 重新连接节点
    def rewire(self, new_node, near_inds):
        nnode = len(self.node_list)
        for i in near_inds:
            near_node = self.node_list[i]
            dx = new_node.x - near_node.x
            dy = new_node.y - near_node.y
            d = math.sqrt(dx ** 2 + dy ** 2)
            scost = new_node.cost + d
            if near_node.cost > scost:
                theta = math.atan2(dy, dx)
                if self.check_collision_extend(new_node, theta, d):
                    near_node.parent = new_node
                    near_node.cost = scost

    # 随机生成节点
    def get_random_node(self):
        # 是否在当前迭代中直接采样目标点，增加搜索空间的多样性，同时提高路径找到目标点的概率
        if random.random() > self.goal_sample_rate:
            rand_node = Node(random.uniform(0, self.map_width), random.uniform(0, self.map_height))
        else:
            rand_node = Node(self.goal.x, self.goal.y)
        return rand_node

    # 启发式生成节点
    def get_informed_random_node(self, c_max):
        if c_max == float("inf"):
            return self.get_random_node()
        c_min = np.linalg.norm([self.start.x - self.goal.x, self.start.y - self.goal.y])
        if c_min == 0:
            return self.get_random_node()
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x ** 2 + y ** 2 <= 1:
                break
        x *= (c_max / 2)
        y *= (math.sqrt(c_max ** 2 - c_min ** 2) / 2)
        theta = math.atan2(self.goal.y - self.start.y, self.goal.x - self.start.x)
        rot_x = x * math.cos(theta) - y * math.sin(theta) + (self.start.x + self.goal.x) / 2
        rot_y = x * math.sin(theta) + y * math.cos(theta) + (self.start.y + self.goal.y) / 2
        return Node(rot_x, rot_y)
    
    # 计算当前路径总成本
    def calc_cost(self, path):
        cost = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            cost += math.sqrt(dx**2 + dy**2)
        return cost

    # 扩展节点
    def steer(self, from_node, to_node, extend_length=float("inf")):
        # 初始化新节点
        new_node = Node(from_node.x, from_node.y)
        # 计算出from-to的方向和距离
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        # 新节点需要加上初始节点的xy方向偏移量
        new_node.x += min(extend_length, d) * math.cos(theta)
        new_node.y += min(extend_length, d) * math.sin(theta)
        # 新节点的当前路径开销
        new_node.cost = from_node.cost + d
        # 新节点的父节点
        new_node.parent = from_node
        return new_node,theta,d

    # 检查节点是否与障碍物碰撞
    def check_collision(self, node):
        if node is None:
            return False
        for ox, oy, size in self.obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            d = math.sqrt(dx ** 2 + dy ** 2)
            if d <= size:
                return False
        if node.x < 0 or node.x > self.map_width or node.y < 0 or node.y > self.map_height:
            return False
        return True

    # 检查扩展的路径是否与障碍物碰撞
    def check_collision_extend(self, near_node, theta, d):
        tmp_node = Node(near_node.x, near_node.y)
        # self.expand_dis / 10可以视情况调整精度
        for i in range(int(10*d//self.expand_dis)):
            tmp_node.x += self.expand_dis / 10 * math.cos(theta)
            tmp_node.y += self.expand_dis / 10 * math.sin(theta)
            if not self.check_collision(tmp_node):
                return False
        return True

    # 计算到目标的距离
    def calc_dist_to_goal(self, x, y):
        dx = x - self.goal.x
        dy = y - self.goal.y
        return math.sqrt(dx ** 2 + dy ** 2)

    # 生成最终路径
    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path

    # 计算两个节点间的距离和角度
    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        return d, theta

    # 找到距离随机节点最近的节点索引
    def get_nearest_node_index(self, node_list, rand_node):
        dlist = [(node.x - rand_node.x) ** 2 + (node.y - rand_node.y) ** 2 for node in node_list]
        min_idx = dlist.index(min(dlist))
        return min_idx

    # 找到新节点附近的节点
    def find_near_nodes(self, new_node):
        nnode = len(self.node_list)
        # 动态范围r
        r = 50.0 * math.sqrt((math.log(nnode) / nnode))
        dlist = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return near_inds

# 绘制圆形障碍物
def plot_circle(x, y, size, color="-k"):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(math.radians(d)) for d in deg]
    yl = [y + size * math.sin(math.radians(d)) for d in deg]
    plt.plot(xl, yl, color)

def main():
    # 定义起点和目标点位置
    start = (0, 0)
    goal = (50, 50)
    # 定义障碍物列表 (x, y, 半径)
    obstacle_list = [
        (5, 5, 4),(10, 10, 3),(15, 15, 2),(8, 22, 4),(18, 24, 2),(25, 25, 2),
        (30, 30, 1),(30, 10, 5),(35, 35, 2),(40, 40, 5),(45, 45, 2),(55, 55, 2),
        (15, 40, 2),(25, 35, 2),(35, 25, 4),(45, 15, 2),(10, 50, 2),(50, 10, 2)
    ]
    # 设置地图尺寸
    map_dims = (60, 60)
    RRT = myRRT(start, goal, map_dims, obstacle_list, expand_dis=1.0,max_iter=1000)

    # 规划路径
    start_time = time.time()
    # path = RRT.rrt_star_planning()
    # path = RRT.informed_rrt_planning()
    path = RRT.dynamic_rrt_planning()
    end_time = time.time()

    # 绘制结果
    if path is None:
        print("Cannot find path")
    else:
        print("Found path!!")
        # 绘制最终路径
        plt.figure()
        for (ox, oy, size) in obstacle_list:
            plot_circle(ox, oy, size)
        for node in RRT.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")
        for (x,y) in path:
            print(x,y)
        print("Time cost: "+str(end_time-start_time)+"s")
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.plot(start[0], start[1], "xr")
        plt.plot(goal[0], goal[1], "xr")
        plt.grid(True)
        plt.axis("equal")
        plt.title("dynamic_rrt Path")
        plt.savefig("dynamic_rrt.png")
        plt.show()

if __name__ == '__main__':
    main()
