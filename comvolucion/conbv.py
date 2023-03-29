import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

H=np.array([[1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1]])
H=H*1/9

gauss=np.array([[1,4,6,4,1],
           [4,16,24,16,4],
           [6,24,36,24,6],
           [4,16,24,16,4],
           [1,4,6,4,1]])

gauss=gauss*1/16
img=Image.open("image.jpg")
imgGray=img.convert("L")
dim=np.shape(imgGray)
Hdim=np.shape(H)
gdim=np.shape(gauss)
F=np.array(imgGray)



final=np.zeros(dim)
for x in range(dim[0]):
  for y in range(dim[1]):
    sum=0
    for i in range(Hdim[0]):
      for j in range(Hdim[1]):
        sum=F[x-i,y-j]*H[i,j]+sum
    final[x,y]=sum


maxs=np.max(final)
img2=final*255/maxs
img2=img2.astype(np.uint8)
fin=Image.fromarray(img2)
fin=fin.convert("L")

finalGauss=np.zeros(dim)
for x in range(dim[0]):
  for y in range(dim[1]):
    sum=0
    for i in range(gdim[0]):
      for j in range(gdim[1]):
        sum=F[x-i,y-j]*gauss[i,j]+sum
    finalGauss[x,y]=sum
    
maxs=np.max(finalGauss)
img3=finalGauss*255/maxs
img3=img3.astype(np.uint8)

fin2=Image.fromarray(img3)
fin2=fin2.convert("L")
#fondo con blur
A = Image.open('A.jpeg')
bg = Image.open('bg.jpeg')
array_A = np.array(A)
array_M = np.array(bg)
array_F = np.array(bg)

Mdim=np.shape(array_M)

f=np.zeros(Mdim)
#fondo blur con gaussiano

for x in range(Mdim[0]):
  for y in range(Mdim[1]):
    sumr=0
    sumg=0
    sumb=0
    for i in range(gdim[0]):
      for j in range(gdim[1]):
        sumr=array_F[x-i,y-j,0]*gauss[i,j]+sumr
        sumg=array_F[x-i,y-j,1]*gauss[i,j]+sumg
        sumb=array_F[x-i,y-j,2]*gauss[i,j]+sumb
    f[x,y,0]=sumr
    f[x,y,1]=sumg
    f[x,y,2]=sumb
maxs=np.max(f)
Fin=f*255/maxs
Fin=Fin.astype(np.uint8)


diff = abs(array_M - array_A)
D = Image.fromarray(diff)
gray_D = D.convert('L')

array_D = np.array(gray_D)
array_U = np.zeros_like(gray_D)
threshold = 150
array_U = np.where(array_D > threshold, 0, 255)             
U = Image.fromarray(array_U)

U=U.convert("RGB")
array_U=np.array(U)

array_R=np.array(array_A)
notU=np.array(array_A)
notU =~array_U
# Imprimir las dimensiones de los arreglos A, F, U y notU para ver si son compatibles
print(array_F.shape,notU.shape, array_A.shape, array_U)
array_R=(array_F&notU)|(array_A&array_U)
R = Image.fromarray(array_R)
plt.subplot(2,3,1)
plt.imshow(imgGray, cmap="gray")
plt.axis("off")
plt.subplot(2,3,2)
plt.imshow(fin,cmap="gray")
plt.axis("off")
plt.subplot(2,3,3)
plt.imshow(fin2,cmap="gray")
plt.axis("off")
plt.subplot(2,3,4)
plt.imshow(A)
plt.axis("off")
plt.subplot(2,3,6)
plt.imshow(R)
plt.axis("off")
plt.show()