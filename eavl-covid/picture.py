import numpy as np

import matplotlib.pyplot as plt

def vap(x, a=-0.6, b=-0.6, c=0.06):
    """ Vapor pressure model """
    return np.exp(a+b/x+c*np.log(x))

def pow3(x, c=0.95, a=0.72, alpha=0.4):
    return c - a * x**(-alpha)

def loglog_linear(x, a=0.23, b=1):
    x = np.log(x)
    return np.log(a*x + b)
#
def dr_hill_zero_background(x, theta=0.7, eta=0.5, kappa=3):
    return (theta* x**eta) / (kappa**eta + x**eta)

def pow4(x, c=1, a=200, b=0.4, alpha=0.1):
    return c - (a*x+b)**-alpha

def mmf(x, alpha=.7, beta=0.01, kappa=0.05, delta=2):

    return alpha - (alpha - beta) / (1. + (kappa * x)**delta)

def exp4(x, c=0.75, a=0.8, b=-0.8, alpha=0.3):
    return c - np.exp(-a*(x**alpha)+b)

def janoschek(x, a=0.73, beta=0.07, k=0.355, delta=0.46):
    """
        http://www.pisces-conservation.com/growthhelp/janoschek.htm
    """
    return a - (a - beta) * np.exp(-k*x**delta)

def weibull(x, alpha=.92, beta=0.02, kappa=0.02, delta=0.5):
    """
    Weibull modell

    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm

    alpha: upper asymptote
    beta: lower asymptote
    k: growth rate
    delta: controls the x-ordinate for the point of inflection
    """
    return alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)

def log_power(x, a=1, b=2.98, c=-0.51):
    #logistic power
    return a/(1.+(x/np.exp(b))**c)

def ilog2(x, c=0.82, a=0.33):
    x = 1 + x
    # assert(np.all(x>1))
    return c - a / np.log(x)

def gompertz(x, a=0.85, b=2.2, c=0.05):
    """
        Gompertz growth function.

        sigmoidal family
        a is the upper asymptote, since
        b, c are negative numbers
        b sets the displacement along the x axis (translates the graph to the left or right)
        c sets the growth rate (y scaling)

        e.g. used to model the growth of tumors

        http://en.wikipedia.org/wiki/Gompertz_function
    """
    return a*np.exp(-b*np.exp(-c*x))
    #return a + b * np.exp(np.exp(-k*(x-i)))

def bertalanffy(x, a=0.88, k=0.02):
    """
        a: asymptote
        k: growth rate
        http://www.pisces-conservation.com/growthhelp/von_bertalanffy.htm
    """
    return a * (1. - np.exp(-k*x))

def data(x, a=0.8, beta=0.07, k=0.355, delta=0.46):
    """
        http://www.pisces-conservation.com/growthhelp/janoschek.htm
    """
    return a - (a - beta) * np.exp(-k*x**delta)

def full_color(x,y,delta,color):
    y1 = y+delta/2
    y2 = y-delta/2
    plt.fill_between(x, y1, y2, color=color, alpha=.4)




x = np.linspace(0, 500, 1000)
# 定义一个线性方程
y1 = vap(x)
full_color(x,y1,0.03,'orange')
y2 = pow3(x)
full_color(x,y2,0.1,'royalblue')

y3 = loglog_linear(x)
full_color(x,y3,0.1,'springgreen')
y4 = dr_hill_zero_background(x)
full_color(x,y4,0.03,'lightpink')
y5 = pow4(x)
full_color(x,y5,0.03,'darkturquoise')
y6 = mmf(x)
full_color(x,y6,0.06,'darkkhaki')

y7 = exp4(x)
full_color(x,y7,0.06,'violet')
y8 = janoschek(x)
full_color(x,y8,0.1,'crimson')

y9 = weibull(x)
full_color(x,y9,0.1,'forestgreen')

y10 = log_power(x)
full_color(x,y10,0.1,'gold')

y11 = ilog2(x)
full_color(x,y11,0.08,'indigo')
y12 = gompertz(x)
full_color(x,y12,0.06,'coral')

y13 = bertalanffy(x)
full_color(x,y13,0.1,'gray')

y14 = data(x)
# 设置x轴的取值范围为：-1到2
plt.xlim(0, 500)
# 设置y轴的取值范围为：-1到3
plt.ylim(0, 1)
# 设置x轴的文本，用于描述x轴代表的是什么
plt.xlabel("Epochs",fontsize = 15)
# 设置y轴的文本，用于描述y轴代表的是什么
plt.ylabel("Accuracy",fontsize = 15)
l2, =plt.plot(x, y2,color = 'royalblue')
# 绘制红色的线宽为1虚线的线条
# plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
l1, = plt.plot(x, y1, color = 'orange')
l3, =plt.plot(x, y3,color = 'springgreen')
l4, =plt.plot(x, y4,color = 'lightpink')
l5, =plt.plot(x, y5,color = 'darkturquoise')
l6, =plt.plot(x, y6,color = 'darkkhaki')
l7, =plt.plot(x, y7,color = 'violet')
l8, =plt.plot(x, y8,color = 'crimson')
l9, =plt.plot(x, y9,color = 'forestgreen')
l10, =plt.plot(x, y10,color = 'gold')
l11, =plt.plot(x, y11,color = 'indigo')
l12, =plt.plot(x, y12,color = 'coral')
l13, =plt.plot(x, y13,color = 'gray')
l14, =plt.plot(x, y14,color = 'black',linewidth=2.5)

new_ticks = np.linspace(0, 500, 6)
plt.xticks(new_ticks)
plt.legend(handles = [l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14],labels = ['vap,$\Delta$y=0.03','pow$_3$,$\Delta$y=0.1','log log linear,$\Delta$y=0.1','Hill$_3$,$\Delta$y=0.1','pow$_4$,$\Delta$y=0.03','mmf,$\Delta$y=0.03','exp$_4$,$\Delta$y=0.06','janoschek,$\Delta$y=0.1','weibull,$\Delta$y=0.1','log power,$\Delta$y=0.08','ilog$_2$,$\Delta$y=0.1','gompertz,$\Delta$y=0.06','bertalanffy,$\Delta$y=0.1','data'],  loc = 'best')
# 显示图表
plt.axvline(x=110,ls="--",c="black")
plt.show()