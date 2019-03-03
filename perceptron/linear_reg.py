import numpy as np
import matplotlib.pyplot as plt

def draw(x1,x2):
    ln=plt.plot(x1,x2,'-')
    plt.pause(0.0001)
    ln[0].remove()
    
def sigmoid(score):
    return 1/(1+np.exp(-score))
    
def error(line_params,points,z):
    m=points.shape[0]
    p=sigmoid(points*line_params)
    cross_entropy=-(np.log(p).T*z+np.log(1-p).T*(1-z))*(1/m)
    return cross_entropy

def gradient_descent(line_params,points,z,alpha):
    for i in range(2000):
        m=points.shape[0]
        p=sigmoid(points*line_params)
        gradient=(points.T*(p-z))*(alpha/m)
        line_params=line_params-gradient
        w1=line_params.item(0)
        w2=line_params.item(1)
        b=line_params.item(2)
        x=np.array([points[:,0].min(),points[:,0].max()])
        #w1x+w2y+b=0
        y=-b/w2+x*(-w1/w2)
        draw(x,y)
        print(error(line_params,all_points,z))

        
    

    
n_pts=100
#random_x_values=np.random.normal(10,2,n_pts)
#random_y_values=np.random.normal(10,2,n_pts)
np.random.seed(0)
bias=np.ones(n_pts)
top_region=np.array([np.random.normal(10,2,n_pts),np.random.normal(12,2,n_pts),bias]).T
bottom_region=np.array([np.random.normal(5,2,n_pts),np.random.normal(6,2,n_pts),bias]).T
all_points=np.vstack((top_region,bottom_region))

line_params=np.matrix([np.zeros(3)]).T
z=np.array([np.zeros(n_pts),np.ones(n_pts)]).reshape(n_pts*2,1)

_, ax=plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0],top_region[:,1],color='r')
ax.scatter(bottom_region[:,0],bottom_region[:,1],color='b')
gradient_descent(line_params,all_points,z,0.06)
plt.show()


