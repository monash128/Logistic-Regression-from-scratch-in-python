import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plotlib

def mapfeature(X1,X2,m):
    features=np.ones([m,1])
    for i in range(1,7):
        for j in range(i+1):
            features=np.c_[features,((X1**(i-j)) * (X2**(j)))]
    return features

def sigmoidFunc(X):
    return (1+np.exp(-X))**(-1)

def hypothesisFunc(X,theta):
    return sigmoidFunc(X @ theta)

def costFuncReg(theta,X,Y,l):
    return ((((-Y).transpose() @ np.log(hypothesisFunc(X,theta))).item() - ((1-Y).transpose() @ np.log(1-hypothesisFunc(X,theta))).item()) + ((l/2) * (np.sum(theta**2))))/m

def gradient(theta,X,Y):
    return (X.transpose() @ (hypothesisFunc(X,theta) - Y))/m

def gradientReg(theta,X,Y,l):
    theta_opt=(X.transpose() @ (hypothesisFunc(X,theta) - Y))/m
    theta_opt[1:]=theta_opt[1:] + (l * theta[1:])/m
    return theta_opt

def plotDecisionBoundary(theta):
    u=np.linspace(-1,1.5,50)
    v=np.linspace(-1,1.5,50)
    z=np.ones([50,50])
    for i in range(50):
        for j in range(50):
            z[i][j]=(mapfeature(u[i],v[j],1) @ theta).item()
    #print(mapped_x.shape)
    #print(z.shape)
    plotlib.contour(u,v,z,0)


data=np.loadtxt("microchipData.txt",delimiter=',')
test_1=data[:,:1]
test_2=data[:,1:2]
(m,p)=test_1.shape
QA_result=data[:,2:3]
plotlib.scatter(test_1[QA_result==1],test_2[QA_result==1],marker="+",label="passed")
plotlib.scatter(test_1[QA_result==0],test_2[QA_result==0],marker="o",label="failed")
plotlib.legend()
#plotlib.show()
test_results=mapfeature(test_1,test_2,m)
(m,k)=test_results.shape

l=1
theta=np.zeros([k,1])
print(costFuncReg(theta,test_results,QA_result,l))
print(gradientReg(theta,test_results,QA_result,l)[:5,:])
print("\n")
test_theta=np.ones([k,1])
test_l=10
print(costFuncReg(test_theta,test_results,QA_result,test_l))
print(gradientReg(test_theta.flatten(),test_results,QA_result.flatten(),test_l)[:5])
#print(test_results.shape,QA_result.shape,theta.shape,theta_optimized.shape)
temp = opt.fmin_tnc(func = costFuncReg,
                    x0 = theta.flatten(),
                    fprime = gradientReg,
                    args = (test_results, QA_result.flatten(),0.1))
theta_optimized = temp[0]
print(theta_optimized[:5])
plotDecisionBoundary(theta_optimized)
plotlib.show()
