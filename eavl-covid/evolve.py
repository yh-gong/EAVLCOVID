# -*- coding: utf-8 -*-


import numpy as np
import heapq
import ga
import os
import file
import shutil
import train
import random
import validation
from conf import settings


class Evolve():

    def __init__(self,GENERATION):

        self.POP_SIZE = settings.POP_SIZE
        self.BATCH_SIZE = settings.BATCH_SIZE
        self.LEARNING_RATE = settings.LEARNING_RATE
        self.NUM_EPOCH = settings.NUM_EPOCH
        self.PROB_CROSSOVER = settings.PROB_CROSSOVER
        self.GENERATION = GENERATION
        self.NUM_BEST_INDIVIDUAL = settings.NUM_BEST_INDIVIDUAL
        self.NUM_CROSSOVER = settings.NUM_CROSSOVER


    def evolve(self):

        generation = self.GENERATION
        index  = 1
        if generation ==1:
            x = ga.GA()
            direction = os.getcwd()
            if os.path.exists('{}/save_data'.format(direction)):
                shutil.rmtree('{}/save_data'.format(direction))
            if os.path.exists(settings.MODEL_PATH):
                shutil.rmtree('{}/{}'.format(direction,settings.MODEL_PATH))
            print('The {} generation began to train====================================>'.format(generation))
            feature, classifier = x.pop(self.POP_SIZE)
            file.write(feature, classifier, self.POP_SIZE, self.GENERATION)
            accrancy = []
            for i in range(self.POP_SIZE):
                acc = train.trainmodel(feature, classifier, generation, i)
                accrancy.append(acc)

        else:
            x = ga.GA()
            # direction = os.getcwd()
            print('The {} generation began to train====================================>'.format(generation))
            # if os.path.exists('{}/save_data/gen_{}/trainingtime'.format(direction, generation)):
            #     shutil.rmtree('{}/save_data/gen_{}/trainingtime'.format(direction,generation))
            feature,classifier = file.read(self.POP_SIZE,generation)
            temp_index = random.randint(self.NUM_BEST_INDIVIDUAL,self.POP_SIZE-1)
            print("Check the {}th population's feature:{}".format(temp_index+1,feature[temp_index]))
            accrancy = []
            accrancy.extend(file.read_acc(generation))
            for i in range(self.POP_SIZE - self.NUM_BEST_INDIVIDUAL):
                temp = i + self.NUM_BEST_INDIVIDUAL
                acc = train.trainmodel(feature, classifier, generation, temp)
                accrancy.append(acc)


        best_acc = max(accrancy)

        print('***********The {} generation best accrancy is : {:.4f}%.***********'.format(generation, 100*best_acc))
        best_acc_list = []
        best_individual_index = list(map(accrancy.index, heapq.nlargest(self.NUM_BEST_INDIVIDUAL, accrancy)))
        for i in range(self.NUM_BEST_INDIVIDUAL):
            temp=[]
            id = best_individual_index[i]
            temp.append(id+1)
            temp.append(accrancy[id])
            best_acc_list.append(temp)
        file.write_acc(best_acc_list,generation)
        # file.move_file(generation,best_individual_index)

        for i in best_individual_index:
            index = file.write_next_pop(feature[i], classifier[i],generation, index)


        fitness  = x.get_fitness(feature, classifier, accrancy)
        index = x.select(fitness, feature, classifier,self.NUM_CROSSOVER,generation,index)

        temp_feature,temp_classifier = x.pop(self.POP_SIZE - self.NUM_BEST_INDIVIDUAL- self.NUM_CROSSOVER)
        for i in range(self.POP_SIZE - self.NUM_BEST_INDIVIDUAL- self.NUM_CROSSOVER):
            index = file.write_next_pop(temp_feature[i],temp_classifier[i],generation,index)



