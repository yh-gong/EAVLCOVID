# -*- coding: utf-8 -*-

import numpy as np
import DNA
import train
import random
from scipy.stats import norm
import file
from conf import settings





class GA():
    def __init__(self):
        self.POP_SIZE = settings.POP_SIZE
        self.BATCH_SIZE = settings.BATCH_SIZE
        self.LEARNING_RATE = settings.LEARNING_RATE
        self.NUM_EPOCH = settings.NUM_EPOCH
        self.PROB_CROSSOVER = settings.PROB_CROSSOVER


    def pop(self,pop_size):
        population_feature = []
        population_classification = []
        for _ in range(pop_size):
            feature, classification = DNA.Chromesome().init_one_individual()
            population_feature.append(feature)
            population_classification.append(classification)

        return population_feature,population_classification


    def train(self,feature, classifier,generation):

        accrancy = train.CNN(self.BATCH_SIZE,self.LEARNING_RATE, self.NUM_EPOCH, self.POP_SIZE, feature, classifier,generation).train()
        return accrancy



    def get_fitness(self,feature, classifier,accrancy):
        F_value = accrancy
        best_individual_id = F_value.index(max(F_value))
        self.best_individual_feature = feature[best_individual_id]
        self.best_individual_classifier = classifier[best_individual_id]
        fitness = F_value - np.min(F_value)
        fitness = fitness ** 2 / (fitness ** 2).sum()
        return fitness

    def select(self,fitness,feature,classifier,num_crossover,GENERATION,index):
        select_index = np.random.choice(self.POP_SIZE,num_crossover,True,fitness)

        temp = 0

        for i in range(int(num_crossover/2)):
            father_feature = feature[select_index[temp]]
            father_classifier = classifier[select_index[temp]]
            temp +=1
            mother_feature = feature[select_index[temp]]
            mother_classifier = classifier[select_index[temp]]
            temp +=1
            child_1_feature, child_2_feature, child_1_classifier, child_2_classifier  = self.crossover(father_feature,mother_feature,father_classifier,mother_classifier,num_crossover)
            index = file.write_next_pop(child_1_feature,child_1_classifier,GENERATION,index)
            index = file.write_next_pop(child_2_feature,child_2_classifier,GENERATION, index)
        return index

    def get_title(self,list):
        return [i[0] for i in list]

    def check_pooling(self,child):

        title = self.get_title(child)
        length = len(title)
        num_pooling = title.count('p')
        child_new = []
        pooling_index = []
        if num_pooling>5:

            for i in range(length):
                if child[i][0] == 'p':
                    pooling_index.append(i)

            for _ in range(num_pooling-5):
                delect_index = random.choice(pooling_index)
                child_new.extend(child[:delect_index])
                child_new.extend(child[delect_index+1:])
                pooling_index.remove(delect_index)
        else:
            child_new =child

        return child_new

    # def check_child(self,child):
    #     child_title = self.get_title(child)
    #     child_len = len(child_title)
    #
    #     for i in range(child_len-1):
    #         index = i+1
    #         temp = index
    #         if (child[index][0]=='c') | (child[index][0]=='cn'):
    #             while child[temp-1][0]== 'p':
    #                 temp -=1
    #             temp -=1
    #
    #             if child[temp][2] != child[index][1]:
    #                 child[index][1] = child[temp][2]
    #
    #     return child



    def check_delect(self,child,best_individual_len):
        child_title = self.get_title(child)
        child_len = len(child_title)
        if child_len - best_individual_len >=5:
            child_new = self.delete(child,child_len - best_individual_len,child_len)
        else:
            child_new =child
        return child_new


    def delete(self,child,length,child_len):
        gauss = norm(loc=length, scale=1)
        prob = []
        for i in range(length+1):
            prob.append(gauss.pdf(i))
        prob = prob / np.array(prob).sum()
        num = int(np.random.choice(length+1,1,True,prob))
        node = random.randint(1,child_len-5)
        child_new = child[:node]
        child_new.extend(child[node+num:])
        return child_new

    def check_add(self,child,best_individual,best_individual_len):

        child_title = self.get_title(child)
        child_len = len(child_title)
        if child_len< (best_individual_len/2):
            start, end = random.sample(range(1, best_individual_len), 2)
            node = random.randint(1, child_len)
            child_new = child[:node]
            child_new.extend(best_individual[start:end])
            child_new.extend(child[node:])
        else:
            child_new = child
        return child_new


    def crossover_classifier(self, father_classifier, mother_classifier):
        len_father_classifier = len(father_classifier)
        len_mother_classifier = len(mother_classifier)
        if len_father_classifier >= len_mother_classifier:
            for i in range(len_mother_classifier):
                temp = mother_classifier[i]
                mother_classifier[i] = father_classifier[i]
                father_classifier[i] = temp
        else:
            for i in range(len_father_classifier):
                temp = father_classifier[i]
                father_classifier[i] = mother_classifier[i]
                mother_classifier[i] = temp

        return father_classifier,mother_classifier

    def crossover(self, father_feature, mother_feature, father_classifier, mother_classifier,num_crossover):
        # classifier = self.classifier
        father_dna = father_feature
        mother_dna = mother_feature
        father_dna_classifier = father_classifier
        mother_dna_classifier = mother_classifier
        if random.random()< self.PROB_CROSSOVER:
            list_father = self.get_title(father_dna)
            list_mother = self.get_title(mother_dna)
            node1 = random.randint(1, len(list_father) - 1)
            node2 = random.randint(1, len(list_mother) - 1)
            child_1_feature = father_dna[:node1]
            child_1_feature.extend(mother_dna[node2:])
            child_2_feature = mother_dna[:node2]
            child_2_feature.extend(father_dna[node1:])
            best_individual = self.best_individual_feature
            best_individual_title = self.get_title(best_individual)
            best_individual_len = len(best_individual_title)
            child_1_feature = self.check_delect(child_1_feature, best_individual_len)
            child_2_feature = self.check_delect(child_2_feature, best_individual_len)
            child_1_feature = self.check_add(child_1_feature, best_individual, best_individual_len)
            child_2_feature = self.check_add(child_2_feature, best_individual, best_individual_len)
            child_1_feature = self.check_pooling(child_1_feature)
            child_2_feature = self.check_pooling(child_2_feature)


            # if len(father_classifier) !=1 & len(mother_classifier) !=1:
            #     child_1_classifier, child_2_classifier = self.crossover_classifier(father_classifier, mother_classifier)
            # else:
            child_1_classifier = father_classifier
            child_2_classifier = mother_classifier
        else:

            child_1_feature = father_feature
            child_2_feature = mother_feature
            child_1_classifier = father_classifier
            child_2_classifier  = mother_classifier


        return child_1_feature, child_2_feature, child_1_classifier , child_2_classifier






# if __name__ == '__main__':
#     test = GA(1,2,3,4)
#     test.best_individual_feature = [['a',1,2,3,4,5,6,7,8,9],
#                                     ['b',2,3,4,5,6,7,8,9,0,1,2]]






