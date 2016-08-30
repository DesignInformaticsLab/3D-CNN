function show_slice(t)
v = t;
n=size(v,1);
[x,y,z] = meshgrid(1:1:n,1:1:n,1:1:n);
xslice = 1;
yslice = 1; 
d = floor(n/2);
zslice = [1,d]; 
slice(x,y,z,v,xslice,yslice,zslice)
colorbar
axis([0,n,0,n,0,n]);
xlabel('x');
ylabel('y');
zlabel('z');
shading interp;
axis([0,size(t,1),0,size(t,2),0,size(t,3)]);
view(2)
