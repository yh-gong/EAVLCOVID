# -*- coding: utf-8 -*-
import os
import csv
import shutil




def write(feature, classifier,POP_SIZE,generation):
    direction = os.getcwd()
    if 1-os.path.exists('{}/save_data'.format(direction)):
        os.makedirs('{}/save_data/'.format(direction))

    os.makedirs('{}/save_data/gen_{}/'.format(direction, generation))

    for i in range(POP_SIZE):
        with open('{}/save_data/gen_{}/pop_{}_feature.csv'.format(direction, generation, i+1), 'w',
                  newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in feature[i]:
                writer.writerow(row)

        with open('{}/save_data/gen_{}/pop_{}_classifier.csv'.format(direction, generation, i + 1), 'w',
                      newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in classifier[i]:
                writer.writerow([row])

def read(POP_SIZE,generation):
    direction  = os.getcwd()
    feature = []
    classifer = []

    for i in range(POP_SIZE):
        with open('{}/save_data/gen_{}/pop_{}_feature.csv'.format(direction, generation, i + 1), 'r') as f:
            reader = csv.reader(f)
            temp=[]


            for row in reader:
                temp_list = []
                for j in row:
                    if (j == 'c') | (j == 'p') |(j== 'cn'):
                        temp_list.append(j)
                    else:
                        temp_list.append(int(j))

                        # temp_list.extend([list(map(int,j))])

                temp.append(temp_list)

        feature.append(temp)

        with open('{}/save_data/gen_{}/pop_{}_classifier.csv'.format(direction, generation, i + 1), 'r') as f:
            reader = csv.reader(f)
            temp = []
            for row in reader:
                temp.append(int(row[0]))
            classifer.append(temp)
    return feature, classifer

def write_next_pop(feature, classifier,GENERATION, index):
    generation = GENERATION + 1
    direction = os.getcwd()

    if 1 - os.path.exists('{}/save_data/gen_{}/'.format(direction, generation)):
        os.makedirs('{}/save_data/gen_{}/'.format(direction, generation))


    with open('{}/save_data/gen_{}/pop_{}_feature.csv'.format(direction, generation, index), 'w',
                  newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in feature:
            writer.writerow(row)

    with open('{}/save_data/gen_{}/pop_{}_classifier.csv'.format(direction, generation, index), 'w',
                  newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in classifier:
            writer.writerow([row])
    return index+1




def write_acc(best_acc,generation):
    direction = os.getcwd()
    with open('{}/save_data/gen_{}/best_acc.csv'.format(direction, generation), 'w',
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in best_acc:
            writer.writerow(row)

def read_acc(generation):
    direction = os.getcwd()
    files = os.listdir('{}/save_data/gen_{}/'.format(direction, generation-1))
    file_name = ''
    for f in files:
        if 'best_acc' in f:
            file_name = f

    dir = '{}/save_data/gen_{}/'.format(direction, generation-1)+file_name
    temp = []
    with open(dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader :
            acc = row
            temp.append(float(acc[1]))

    return temp

def write_time(time,generation,index):
    direction = os.getcwd()
    path = '{}/save_data/gen_{}/trainingtime/'.format(direction, generation)
    if 1-os.path.exists(path):
        os.makedirs(path)

    with open('{}/save_data/gen_{}/trainingtime/pop_{}.csv'.format(direction, generation, index), 'w',
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in time:
            writer.writerow([row])

# def move_file(generation,best_acc_index):
#     direction = os.getcwd()
#     path1 = '{}/save_data/gen_{}/trainingtime/'.format(direction, generation)
#     path2 = '{}/save_data/gen_{}/trainingtime/'.format(direction, generation-1)
#     if 1-os.path.exists(path2):
#         os.makedirs(path2)
#
#     for _,_,filenames in os.walk(path1):
#         for filename in filenames:
#             for id,index in enumerate (best_acc_index):
#                 if filename == 'pop_{}'.format(index):
#                     new_name = 'pop_{}'.format(id+1)
#                     shutil.copyfile(os.path.join(path2,filename),os.path.join(path1,new_name))




if __name__ == '__main__':
    feature = [[['c', 1, 2, 3, 4],
                ['d', 5, 6, 7, 8],
                ['e', 9, 10, 11, 12],
                ['f', 13, 14, 15, 16],
                ['cc', 1, 2, 3, 4, 5, 6, 7, 8, 8],
                ['dd', 1, 2, 3, 4, 5, 3, 2, 4, 52],
                ['ee', 3, 2, 4, 5, 7, 4, 3, 2, 45, 65, 76],
                ['ff', 2, 4, 6, 7, 3, 2, 5, 6, 7, 4, 3]],
               [['g', 17, 18, 19, 20],
                ['h', 21, 22, 23, 24],
                ['i', 25, 26, 27, 28],
                ['j', 29, 30, 31, 32],
                ['gg', 23, 3, 4, 5, 76, 3, 2, 2],
                ['hh', 2, 3, 4, 5, 67, 45, 3],
                ['ii', 2, 3, 324, 52, 334, 2, 3, 4, 2],
                ['jj', 5, 5, 3, 45, 4, 43, 2]],
               [['c', 17, 18, 19, 20],
                ['p', 21, 22, 23, 24],
                ['p', 25, 26, 27, 28],
                ['p', 29, 30, 31, 32],
                ['p', 23, 3, 4, 5, 76, 3, 2, 2],
                ['p', 2, 3, 4, 5, 67, 45, 3],
                ['p', 2, 3, 324, 52, 334, 2, 3, 4, 2],
                ['p', 5, 5, 3, 45, 4, 43, 2]]
               ]

    classifer = [[124],
                  [12345,23434,23422],
                  [1232,2323]]
    read(3,2)
