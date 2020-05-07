from collections import deque
import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        super(ReplayBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.count = 0
        #self.s_p0 = deque()
        #self.s_pk = deque()
        #self.s_free_space = deque()
        #self.s_path = deque()
        #self.action = deque()
        #self.reward = deque()
        #self.terminal = deque()
        #self.s2_p0 = deque()
        #self.s2_pk = deque()
        #self.s2_free_space = deque()
        #self.s2_path = deque()


    def save(self, experience):
        if self.count == 0:
            self.s_p0 = np.array(experience[0][0])
            self.s_pk = np.array(experience[0][1])
            self.s_free_space = np.array(experience[0][2])
            self.s_path = np.array(experience[0][3])
            self.action = np.array(experience[1])
            self.reward = np.array([experience[2]])
            self.terminal = np.array([[experience[3]]])
            self.s2_p0 = np.array(experience[4][0])
            self.s2_pk = np.array(experience[4][1])
            self.s2_free_space = np.array(experience[4][2])
            self.s2_path = np.array(experience[4][3])

        if self.count < self.buffer_size:
            self.count += 1
        else:
            for x in [self.s_p0, self.s_pk, self.s_free_space, self.s_path, self.action, self.reward, self.terminal,
                      self.s2_p0, self.s2_pk, self.s2_free_space, self.s2_path]:
                np.delete(x, 0, axis=0)

        self.s_p0 = np.concatenate([self.s_p0, experience[0][0]], axis=0)
        self.s_pk = np.concatenate([self.s_pk, experience[0][1]], axis=0)
        self.s_free_space = np.concatenate([self.s_free_space, experience[0][2]], axis=0)
        self.s_path = np.concatenate([self.s_path, experience[0][3]], axis=0)
        self.action = np.concatenate([self.action, experience[1]], axis=0)
        self.reward = np.concatenate([self.reward, experience[2][np.newaxis]], axis=0)
        self.terminal = np.concatenate([self.terminal, experience[3][np.newaxis, np.newaxis]], axis=0)
        self.s2_p0 = np.concatenate([self.s2_p0, experience[4][0]], axis=0)
        self.s2_pk = np.concatenate([self.s2_pk, experience[4][1]], axis=0)
        self.s2_free_space = np.concatenate([self.s2_free_space, experience[4][2]], axis=0)
        self.s2_path = np.concatenate([self.s2_path, experience[4][3]], axis=0)

        #if self.count < self.buffer_size:
        #    self.count += 1
        #else:
        #    for x in [self.s_p0, self.s_pk, self.s_free_space, self.s_path, self.action, self.reward, self.terminal,
        #              self.s2_p0, self.s2_pk, self.s2_free_space, self.s2_path]:
        #        x.popleft()
        #self.s_p0.append(experience[0][0])
        #self.s_pk.append(experience[0][1])
        #self.s_free_space.append(experience[0][2])
        #self.s_path.append(experience[0][3])
        #self.action.append(experience[1])
        #self.reward.append(experience[2])
        #self.terminal.append(experience[3])
        #self.s2_p0.append(experience[4][0])
        #self.s2_pk.append(experience[4][1])
        #self.s2_free_space.append(experience[4][2])
        #self.s2_path.append(experience[4][3])

    def size(self):
        return self.count

    def get_batch(self, batch_size):
        idx = random.sample(range(self.count), batch_size)
#
#        l = [[], [], [], [], [], [], [], [], [], [], []]
#        for idx in idxs:
#            l[0] = self.s_p0[idx]
#            l[1].append(self.s_pk[idx])
#            l[2].append(self.s_free_space[idx])
#            l[3].append(self.s_path[idx])
#            l[4].append(self.action[idx])
#            l[5].append(self.reward[idx])
#            l[6].append(self.terminal[idx])
#            l[7].append(self.s2_p0[idx])
#            l[8].append(self.s2_pk[idx])
#            l[9].append(self.s2_free_space[idx])
#            l[10].append(self.s2_path[idx])
#
#        np_l = []
#        for x in l:
#            np_l.append(np.concatenate([x], axis=0))

        return (self.s_p0[idx], self.s_pk[idx], self.s_free_space[idx], self.s_path[idx]), \
               self.action[idx], self.reward[idx], self.terminal[idx], \
               (self.s2_p0[idx], self.s2_pk[idx], self.s2_free_space[idx], self.s2_path[idx])
