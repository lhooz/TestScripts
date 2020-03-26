# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:02:02 2019

@author: Hao
"""

from numpy import zeros,array,sqrt,matmul,einsum,eye,where,delete,reshape,insert,concatenate
from numpy.linalg import norm,det,inv
from math import sin,pi
import matplotlib.pyplot as plt

Cbasis=eye(2)

def LagP1 (x): 
    """Laglangian polynomial of order 1"""
    
    result=zeros([2,2])
    
    N1=(1-x)/2
    N2=(1+x)/2
    
    N1x=-1/2
    N2x=1/2
    
    resultSF=[N1,N2]
    resultGd=[N1x,N2x]
    
    result[0]=resultSF
    result[1]=resultGd
    
    return result

def Quadrilateral (x,y):
    """Quadrilateral element"""
    
    result=[]
    resultSF=zeros([4])
    resultGd=zeros([2,4])
    
    resultSF=[LagP1(x)[0][0]*LagP1(y)[0][0],LagP1(x)[0][1]*LagP1(y)[0][0],\
              LagP1(x)[0][1]*LagP1(y)[0][1],LagP1(x)[0][0]*LagP1(y)[0][1]]
    
    resultGd[0]=[LagP1(x)[1][0]*LagP1(y)[0][0], LagP1(x)[1][1]*LagP1(y)[0][0],\
                 LagP1(x)[1][1]*LagP1(y)[0][1], LagP1(x)[1][0]*LagP1(y)[0][1]]
    
    resultGd[1]=[LagP1(x)[0][0]*LagP1(y)[1][0], LagP1(x)[0][1]*LagP1(y)[1][0],\
                 LagP1(x)[0][1]*LagP1(y)[1][1], LagP1(x)[0][0]*LagP1(y)[1][1]]
    
    result.append(resultSF)
    result.append(resultGd)
    
    return result

def barmesh (l,h,n):
    """mesh of stright bar"""
    
    Nodes=[]
    Nodesb=[]
    element=[]
    elementnb=[]
    grid=[]
    Node=0;
    
    for i in range(n):
        Nodei1=[l/n*i,0]
        Nodei2=[l/n*(i+1),0]
        Nodei3=[l/n*(i+1),h]
        Nodei4=[l/n*i,h]
        
        Nodes.append([Nodei1,Nodei2,Nodei3,Nodei4])
        
        Nodesb.append([i,[0,-1]])
        Nodesb.append([i,[0,1]])
        if i==n-1:
            Nodesb.append([i,[1,0]])
    #breakpoint()
    for i in range(len(Nodes)):
        elementi=[]
        for j in range(4):
            NID=checkNode(Nodes[i][j],grid)
            if (NID==0):
                NID=Node
                grid.append([Nodes[i][j][0],Nodes[i][j][1]])
                Node +=1
            
            elementi.append(NID)
            
        element.append(elementi)
        
    for i in range(len(Nodesb)):
        elementi=[]
        Ni=array(Nodesb[i][1])
        elementi.append(Nodesb[i][0])
        elementi.append(Nodesb[i][1])
        elementi.append(Ni)
        elementnb.append(elementi)
        
    def fEsb(t):
        """Essential boundary function, node No in ascending order"""
        
        result=[[0,0,0],\
                [3,0,0]]
        
        return result
    
    def fNab(t):
        """Natural boundary function"""
        
        resultu=[]
        results=[]
        resultl=[]
        
        for i in range(len(element)):
            elem=element[i]
            resultu.append([elem[3],1000*sin(2*pi*0*t)-1000])
            resultl.append([elem[0],1000*sin(2*pi*0*t)-1000])
            if i==len(element)-1:
                results.append([elem[1],1000*sin(2*pi*0*t)+1000])
                results.append([elem[2],1000*sin(2*pi*0*t)+1000])
            
        resultu.reverse()
        result=resultl+results+resultu
        
        return result
    
    mesh=[grid,element,elementnb,fEsb,fNab]
    
    return mesh

def checkNode (x,grid):
    """Ckeck Node ID for coincidence"""
    
    if(len(grid)>=1):
        Node=0
        #breakpoint()
        for b in grid:
            y=array(b)
            if (norm(x-y)<1e-1):
                return Node
            
            Node +=1
            
    return 0

def Jabob(mesh,eleFunc):
    """Array of Jacobian of geometric mapping for each element"""
    
    Jacob=[]
    
    for elem in mesh[1]:
        EleGeoCo=array([mesh[0][elem[0]],mesh[0][elem[1]],\
                        mesh[0][elem[2]],mesh[0][elem[3]]])
        #print(EleGeoCo,sep="\n")
        def Jfunc(x,y):
            Jacobi=matmul(eleFunc(x,y)[1],EleGeoCo).T#Jacobian of geometric mapping
            return Jacobi
        #print(matmul(eleFunc(0,1)[1],EleGeoCo).T,sep="\n")
        
        Jacob.append(Jfunc)
    return Jacob

def DefGradFU (mesh,eleFunc,disp):
    """Array of deformation gradient for each element"""
    
    Grad=[]
    
    for i in range(len(mesh[1])):
        elem=mesh[1][i]
        #breakpoint()
        def Gdfunc(x,y):
            Uj=0
            Jacobi=Jabob(mesh,eleFunc)[i](x,y)
            for j in elem:
                Gj=matmul(array([eleFunc(x,y)[1].T[elem.index(j)]]),inv(Jacobi))#Gradient of basis function for jth Node
                #print(Gj,sep="\n")
                Uj=Uj+matmul(array([disp[j]]).T,Gj)#Displacement gradient function for jth Node
                #print(Uj,sep="\n")
            Fj=eye(2)+Uj#Deformation gradient function for jth Node
            #print(Fj,sep="\n")
            Gradi=[Fj,Uj]
            #breakpoint()
            return Gradi
        
        Grad.append(Gdfunc)
    return Grad

def InertiaMx (mesh,eleFunc,dens):
    """Inertia matrix of problem"""
    
    dim=len(mesh[0])
    Mmx=zeros([2*dim,2*dim])
    
    for k in range(len(mesh[1])):
        elem=mesh[1][k]
        for i in elem:
            for j in elem:
                def Ifunc(x,y,basisi):
                    """Function to integrate for inertia term"""
                    Jacobi=Jabob(mesh,eleFunc)[k](x,y)
                    #breakpoint()
                    result=eleFunc(x,y)[0][elem.index(i)]*\
                           eleFunc(x,y)[0][elem.index(j)]*\
                           det(Jacobi)
                    if i in array(mesh[3](0))[:,0]:
                        result=0
                    #breakpoint()
                    return result
                
                Mmx[2*i,2*j]=Mmx[2*i,2*j]+GQ(Ifunc,0)*dens
                Mmx[2*i+1,2*j+1]=Mmx[2*i+1,2*j+1]+GQ(Ifunc,1)*dens
    
    return Mmx

def ConstitutiveVar(mesh,eleFunc,Ctensor,disp):
    """Constitutive Variation of problem"""
    
    dim=len(mesh[0])
    CVar=zeros([2*dim])
    
    for k in range(len(mesh[1])):
        elem=mesh[1][k]
        for i in elem:
            def Cfunc(x,y,basisi):
                """Function to integrate for constitutive term"""
                Jacobi=Jabob(mesh,eleFunc)[k](x,y)
                Grad=DefGradFU(mesh,eleFunc,disp)[k]
                GradF=Grad(x,y)[0]
                GradU=Grad(x,y)[1]
                #print(GradU,sep="\n")
                #breakpoint()
                Gi=matmul(array([eleFunc(x,y)[1].T[elem.index(i)]]),inv(Jacobi))#Gradient of basis function for ith Node
                S=einsum('...ij,ij',Ctensor,0.5*(matmul(GradU.T,GradU)+GradU.T+GradU))#Second Piola-Kirchhoff stress tensor
                P=matmul(GradF,S)#First Piola-Kirchhoff stress tensor
                #print(P,sep="\n")
                #breakpoint()
                result=einsum('ij,ij',matmul(array([Cbasis[basisi]]).T,Gi),P)*det(Jacobi)
                #print(result,sep="\n")
                if i in array(mesh[3](0))[:,0]:
                    result=0
                #breakpoint()
                return result
                
            CVar[2*i]=CVar[2*i]+GQ(Cfunc,0)
            CVar[2*i+1]=CVar[2*i+1]+GQ(Cfunc,1)
    
    return CVar

def LinConstitutiveMx(mesh,eleFunc,Ctensor,disp):
    """Linear constitutive matrix for tangent stiffness"""
    
    dim=len(mesh[0])
    Kmx=zeros([2*dim,2*dim])
    
    for k in range(len(mesh[1])):
        elem=mesh[1][k]
        for i in elem:
            for j in elem:
                def Kfunc1(x,y,basisi):
                    """Function to integrate for tangent stiffness matrix"""
                    Jacobi=Jabob(mesh,eleFunc)[k](x,y)
                    Grad=DefGradFU(mesh,eleFunc,disp)[k]
                    GradF=Grad(x,y)[0]
                    GradU=Grad(x,y)[1]
                    Gi=matmul(array([eleFunc(x,y)[1].T[elem.index(i)]]),inv(Jacobi))#Gradient of basis function for ith Node
                    Gj=matmul(array([eleFunc(x,y)[1].T[elem.index(j)]]),inv(Jacobi))#Gradient of basis function for jth Node
                    #print(Gj,sep="\n")
                    S=einsum('...ij,ij',Ctensor,0.5*(matmul(GradU.T,GradU)+GradU.T+GradU))#Second Piola-Kirchhoff stress tensor
                    Dtensor=einsum('ij,jklm,nl->iknm',GradF,Ctensor,GradF,optimize='greedy')+einsum('ij,kl->ikjl',eye(2),S)#Tangent stiffness tensor
                    result=einsum('ij,ijkl,l->k',matmul(array([Cbasis[basisi]]).T,Gi),Dtensor,Gj[0])[0]*det(Jacobi)
                    #print(result,sep="\n")
                    if i in array(mesh[3](0))[:,0]:
                        result=0
                #breakpoint()
                    return result
                def Kfunc2(x,y,basisi):
                    """Function to integrate for tangent stiffness matrix"""
                    Jacobi=Jabob(mesh,eleFunc)[k](x,y)
                    Grad=DefGradFU(mesh,eleFunc,disp)[k]
                    GradF=Grad(x,y)[0]
                    GradU=Grad(x,y)[1]
                    Gi=matmul(array([eleFunc(x,y)[1].T[elem.index(i)]]),inv(Jacobi))#Gradient of basis function for ith Node
                    Gj=matmul(array([eleFunc(x,y)[1].T[elem.index(j)]]),inv(Jacobi))#Gradient of basis function for jth Node
                    S=einsum('...ij,ij',Ctensor,0.5*(matmul(GradU.T,GradU)+GradU.T+GradU))#Second Piola-Kirchhoff stress tensor
                    Dtensor=einsum('ij,jklm,nl->iknm',GradF,Ctensor,GradF,optimize='greedy')+einsum('ij,kl->ikjl',eye(2),S)#Tangent stiffness tensor
                    result=einsum('ij,ijkl,l->k',matmul(array([Cbasis[basisi]]).T,Gi),Dtensor,Gj[0])[1]*det(Jacobi)
                    #print(result,sep="\n")
                    if i in array(mesh[3](0))[:,0]:
                        result=0
                #breakpoint()
                    return result
                
                Kmx[2*i,2*j]=Kmx[2*i,2*j]+GQ(Kfunc1,0)
                Kmx[2*i+1,2*j]=Kmx[2*i+1,2*j]+GQ(Kfunc1,1)
                Kmx[2*i,2*j+1]=Kmx[2*i,2*j+1]+GQ(Kfunc2,0)
                Kmx[2*i+1,2*j+1]=Kmx[2*i+1,2*j+1]+GQ(Kfunc2,1)
    
    return Kmx

def SurfaceTVar (mesh,eleFunc,disp,p):
    """Surface traction variation of problem"""
    
    dim=len(mesh[0])
    STmx=zeros([2*dim,dim])
    
    for k in range(len(mesh[2])):
        eleNo=mesh[2][k][0]
        elem=mesh[1][eleNo]
        #breakpoint()
        for i in elem:
            for j in elem:
                def SfuncO(x,y,basisi):
                    """Function to integrate for Surface traction term"""
                    Jacobi=Jabob(mesh,eleFunc)[eleNo](x,y)
                    Grad=DefGradFU(mesh,eleFunc,disp)[eleNo]
                    GradF=Grad(x,y)[0]
                    #breakpoint()
                    #print(basisi,sep="\n")
                    result=eleFunc(x,y)[0][elem.index(i)]*eleFunc(x,y)[0][elem.index(j)]*det(GradF)*det(Jacobi)*\
                           einsum('ij,jk,k->i',inv(GradF).T,inv(Jacobi).T,mesh[2][k][2])[basisi]
                           
                    if i in array(mesh[3](0))[:,0]:
                        result=0
                    #breakpoint()
                    return result
                
                def Sfunc(x,basisi):
                    if mesh[2][k][1][0]==0:
                        return SfuncO(x,mesh[2][k][1][1],basisi)
                    else:
                        return SfuncO(mesh[2][k][1][0],x,basisi)
                    
                STmx[2*i,j]=STmx[2*i,j]+GQ1(Sfunc,0)
                STmx[2*i+1,j]=STmx[2*i+1,j]+GQ1(Sfunc,1)
                
    SVar=matmul(STmx,p)
    
    return SVar

def LinSurfTractionMx (mesh,eleFunc,disp,p):
    """Linear surface traction matrix for surface variation"""
    
    dim=len(mesh[0])
    Tmx=zeros([2*dim,2*dim])
    
    for k in range(len(mesh[2])):
        eleNo=mesh[2][k][0]
        elem=mesh[1][eleNo]
        #breakpoint()
        for i in elem:
            for j in elem:
                def SfuncM(x,y,basisi):
                    """Function to integrate for Surface traction matrix"""
                    Jacobi=Jabob(mesh,eleFunc)[eleNo](x,y)
                    Grad=DefGradFU(mesh,eleFunc,disp)[eleNo]
                    GradF=Grad(x,y)[0]
                    invF=einsum('ij,kl->ijkl',inv(GradF).T,inv(GradF).T)
                    DFTensor=invF-einsum('kjil',invF)
                    Gj=matmul(array([eleFunc(x,y)[1].T[elem.index(j)]]),inv(Jacobi))#Gradient of basis function for jth Node
                    #breakpoint()
                    #print(basisi,sep="\n")
                    def Pfunc(x,y):
                        """Interpolated pressure function within elem"""
                        Pfunc=0
                        for l in elem:
                            Pfunc=Pfunc+eleFunc(x,y)[0][elem.index(l)]*p[l]
                        return Pfunc
                    
                    result=eleFunc(x,y)[0][elem.index(i)]*Pfunc(x,y)*det(GradF)*det(Jacobi)*\
                           einsum('l,ijkl,km,m,i->j',array(Cbasis[basisi]),DFTensor,inv(Jacobi).T,mesh[2][k][2],Gj[0],optimize='greedy')
                           
                    if i in array(mesh[3](0))[:,0]:
                        result=array([0,0])
                    #breakpoint()
                    return result
                
                def Sfunc1(x,basisi):
                    if mesh[2][k][1][0]==0:
                        return SfuncM(x,mesh[2][k][1][1],basisi)[0]
                    else:
                        return SfuncM(mesh[2][k][1][0],x,basisi)[0]
                    
                def Sfunc2(x,basisi):
                    if mesh[2][k][1][0]==0:
                        return SfuncM(x,mesh[2][k][1][1],basisi)[1]
                    else:
                        return SfuncM(mesh[2][k][1][0],x,basisi)[1]
                    
                Tmx[2*i,2*j]=Tmx[2*i,2*j]+GQ1(Sfunc1,0)
                Tmx[2*i+1,2*j]=Tmx[2*i+1,2*j]+GQ1(Sfunc1,1)
                Tmx[2*i,2*j+1]=Tmx[2*i,2*j+1]+GQ1(Sfunc2,0)
                Tmx[2*i+1,2*j+1]=Tmx[2*i+1,2*j+1]+GQ1(Sfunc2,1)
    
    return Tmx

def MaterialModel (E,v):
    """Define material St.Venant-Kirchhoff model based on E and v"""
    
    niu=0.5*E/(1+v)
    namda=v*E/((1+v)*(1-2*v))
    Kronecker=einsum('ij,kl->ijkl',eye(2),eye(2))
    
    Ctensor=namda*Kronecker+niu*(einsum('ikjl',Kronecker)+einsum('iljk',Kronecker))
    
    return Ctensor

def GQ (func,i):
    """Integration of function in [-1,1]^2 using 3 point gaussian quadrature
       exact up to order 5"""
    
    n=-sqrt(3/5)
    p=sqrt(3/5)
    
    rn=(5/9)*(func(n,n,i)+func(p,n,i))+(8/9)*func(0,n,i)
    r0=(5/9)*(func(n,0,i)+func(p,0,i))+(8/9)*func(0,0,i)
    rp=(5/9)*(func(n,p,i)+func(p,p,i))+(8/9)*func(0,p,i)
    
    result=(5/9)*(rn+rp)+(8/9)*r0
    
    return result

def GQ1 (func,i):
    """Integration of function in [-1,1] using 3 point gaussian quadrature
       exact up to order 5"""
    
    n=-sqrt(3/5)
    p=sqrt(3/5)
    
    result=(5/9)*(func(n,i)+func(p,i))+(8/9)*func(0,i)
    
    return result

def Sol1step (mesh,M,eleFunc,Ctensor,xn,p,t):
    """Equilibrium solution for one time step"""
    
    NNodes=len(mesh[0])
    
    ddisp=reshape(xn,(-1,2))[:NNodes,:]
    disp=reshape(xn,(-1,2))[NNodes:,:]
    
    EsbNode=array(mesh[3](t))[:,0]
    
    Delind=[]#Boundary nodes ID to be deleted from mass matrix
    for i in EsbNode:
        Delind.append(2*i)
        Delind.append(2*i+1)
    
    resC=ConstitutiveVar(mesh,eleFunc,Ctensor,disp)
    #print(resC,sep="\n")
    #breakpoint()
    #K=LinConstitutiveMx(mesh,eleFunc,Ctensor,disp)
    resS=SurfaceTVar(mesh,eleFunc,disp,p)
    #print(resS,sep="\n")
    #breakpoint()
    #T=LinSurfTractionMx(mesh,eleFunc,disp,p)
    res=resS-resC
    
    M=delete(M,Delind,0)
    Mg=M[:,Delind]
    M=delete(M,Delind,1)
    
    gc=array(mesh[3](t))[:,1:]
    gc=reshape(gc,-1)
    #breakpoint()
    res=delete(res,Delind,0)-matmul(Mg,gc)
    
    a=matmul(inv(M),res)
    a=reshape(a,(-1,2))
    for i in range(len(EsbNode)):
        a=insert(a,EsbNode[i],array(mesh[3](t))[i,1:],axis=0)
    
    dxn=concatenate((a,ddisp),axis=None)
    return dxn

def rk4(disp,ddisp,mesh,M,Quadrilateral,Ctensor,p,t,dt):
    """4th order Runge Kutta method to solve ODE"""
    
    NNodes=len(mesh[0])
    
    xn=concatenate((ddisp,disp),axis=None)
    #breakpoint()
    k1=dt*Sol1step(mesh,M,Quadrilateral,Ctensor,xn,p,t)
    #print(k1,sep="\n")
    #breakpoint()
    k2=dt*Sol1step(mesh,M,Quadrilateral,Ctensor,xn+0.5*k1,p,t+0.5*dt)
    #print(k2,sep="\n")
    #breakpoint()
    k3=dt*Sol1step(mesh,M,Quadrilateral,Ctensor,xn+0.5*k2,p,t+0.5*dt)
    #print(k3,sep="\n")
    #breakpoint()
    k4=dt*Sol1step(mesh,M,Quadrilateral,Ctensor,xn+k3,p,t+dt)
    
    xn1=xn+(1/6)*(k1+2*k2+2*k3+k4)
    
    ddisp1=reshape(xn1,(-1,2))[:NNodes,:]
    disp1=reshape(xn1,(-1,2))[NNodes:,:]
    
    return [disp1,ddisp1]

if __name__== "__main__":
    
    dt=3e-4#Time step
    
    mesh=barmesh(1,1,1)
    Ctensor=MaterialModel(7e10,0.3)
    density=2700
    disp=zeros([len(mesh[0]),2])
    ddisp=zeros([len(mesh[0]),2])
    N=5#No of time steps
    
    x=zeros([len(mesh[0]),N])
    y=zeros([len(mesh[0]),N])
    
    for n in range(N):
        """Time marching solution"""
        t=n*dt
        """Obtain list of intepolated Natural boundary loads on each Node"""
        p=[]
        NbNodes=array(mesh[4](t)).T[0]
        Pi=array(mesh[4](t)).T[1]
        for i in range(len(mesh[0])):
            indi=where(NbNodes==i)[0]
            if len(indi)==0:
                p.append(0)
            else:
                p.append(Pi[indi[0]])
        M=InertiaMx(mesh,Quadrilateral,density)#Mass matrix of problem
        
        soln=rk4(disp,ddisp,mesh,M,Quadrilateral,Ctensor,p,t,dt)
        
        disp=soln[0]
        ddisp=soln[1]
        
        Grad=DefGradFU(mesh,Quadrilateral,disp)[0]
        GradF=Grad(1,0)[0]
        GradU=Grad(1,0)[1]
        S=einsum('...ij,ij',Ctensor,0.5*(matmul(GradU.T,GradU)+GradU.T+GradU))
        #print(S,sep="\n")
        #breakpoint()
        resC=ConstitutiveVar(mesh,Quadrilateral,Ctensor,disp)
        resS=SurfaceTVar(mesh,Quadrilateral,disp,p)
        print(resC,sep="\n")
        breakpoint()
        print(resS,sep="\n")
        breakpoint()
        
        xyt=mesh[0]+disp
        x[:,n]=xyt[:,0]
        y[:,n]=xyt[:,1]
        
    for nt in range(N):
        xt=[]
        yt=[]
        for nb in NbNodes:
            xt.append(x[int(nb)][nt])
            yt.append(y[int(nb)][nt])
        
        plt.plot(xt,yt)
    #for i in range(len(mesh[1])):
        #print(Jabob(mesh,Quadrilateral)[i](1,0),sep="\n")
        #print(DefGradFU(mesh,Quadrilateral,disp)[i](0,0)[1],sep="\n")
    
    #print(*res1, sep= "\n")
