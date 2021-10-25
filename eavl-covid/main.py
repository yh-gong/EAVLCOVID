import ga
import evolve
from conf import settings




if __name__ =='__main__':
    flag = 1
    GENERATION = 21
    while flag == 1:
        evolve.Evolve(GENERATION).evolve()
        GENERATION +=1



