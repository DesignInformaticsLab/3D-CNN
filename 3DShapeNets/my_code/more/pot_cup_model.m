% visualize artificial pot and cup training model

figure;

subplot(2,2,4)
load('/home/hope-yao/Dropbox/mesh convert/more/cup_train.mat')
the_sample=instance;threshold=0.1;
p = patch(isosurface(the_sample,threshold));
set(p,'FaceColor','red','EdgeColor','none');
daspect([1,1,1])
grid on; axis tight

camlight
lighting gouraud;
set(gcf,'Color','white');
% set(gca,'position',[0,0,1,1],'units','normalized');
axis equal
% % axis vis3d
axis([0 30 0 30 0 30])
view(2);
title('cup train');

subplot(2,2,3)
load('/home/hope-yao/Dropbox/mesh convert/more/cup_test.mat')
the_sample=instance;threshold=0.1;
p = patch(isosurface(the_sample,threshold));
set(p,'FaceColor','red','EdgeColor','none');
daspect([1,1,1])
grid on; axis tight

camlight
lighting gouraud;
set(gcf,'Color','white');
% set(gca,'position',[0,0,1,1],'units','normalized');
axis equal
% % axis vis3d
axis([0 30 0 30 0 30])
view(2);
title('cup test');


subplot(2,2,2)
load('/home/hope-yao/Dropbox/mesh convert/more/pot_test_new.mat')
the_sample=instance;threshold=0.1;
p = patch(isosurface(the_sample,threshold));
set(p,'FaceColor','red','EdgeColor','none');
daspect([1,1,1])
grid on; axis tight

camlight
lighting gouraud;
set(gcf,'Color','white');
% set(gca,'position',[0,0,1,1],'units','normalized');
axis equal
% % axis vis3d
axis([0 30 0 30 0 30])
view(2);
title('pot test');



subplot(2,2,1)
load('/home/hope-yao/Dropbox/mesh convert/more/pot_train_new.mat')
the_sample=instance;threshold=0.1;
p = patch(isosurface(the_sample,threshold));
set(p,'FaceColor','red','EdgeColor','none');
daspect([1,1,1])
grid on; axis tight

camlight
lighting gouraud;
set(gcf,'Color','white');
% set(gca,'position',[0,0,1,1],'units','normalized');
axis equal
% % axis vis3d
axis([0 30 0 30 0 30])
view(2);
title('pot train');
