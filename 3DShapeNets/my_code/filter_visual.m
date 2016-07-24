function filter_visual()
load('./pretrained_model.mat');
layer_idx = 2; % n-th hidden layer
w = rec_conv(model,layer_idx,[]);
save('w_projected.mat','w');
plot_filter(w)
show_sample(w,0.5)
end

function plot_filter(w)

figure;
assert(size(w,1)>=6);
for i=1:6
    subplot(2,3,i)
    vol3d('cdata', squeeze(w(i,:,:,:)), 'xdata', [0 1], 'ydata', [0 1], 'zdata', [0 1]);
    colormap(bone(256));
    alphamap([0 linspace(0.1, 0, 2)]);
    %         axis equal off
%     axis([0.1,0.9,0.1,0.9,0.1,0.9])
    set(gcf, 'color', 'w');
    view(3);
end

end


function w = rec_conv(model,layer_idx,mat2)

if isempty(mat2)
    mat2 = squeeze(model.layers{layer_idx+1}.w); % upper layer
end
mat1 = squeeze(model.layers{layer_idx}.w); % lower layer
l1 = size(mat1,2);
l2 = size(mat2,2);
w = zeros(size(mat2,1),l1+l2-1,l1+l2-1,l1+l2-1,size(mat1,5));
for i=1:size(mat2,1) % number of final figures
    for j=1:size(mat2,5) % number of channels
        for k=1:size(mat1,5)
            w1 = squeeze(mat1(j,:,:,:,k));
            w2 = squeeze(mat2(i,:,:,:,j));
            w(i,:,:,:,k) = squeeze(w(i,:,:,:,k)) + convn(w2,w1);
        end
    end
end
layer_idx = layer_idx - 1;
if(layer_idx>2)
    w = rec_conv(model,layer_idx,w);
else
    w = rec_conv2(model.layers{2}.w,w);
    return;
end

end

function w = rec_conv2(mat1,mat2)
l1 = size(mat1,2);
l2 = size(mat2,2);
w = zeros(size(mat2,1),l1+l2-1,l1+l2-1,l1+l2-1);
for i=1:size(mat2,1) % number of channels
    for j=1:size(mat2,5)
        w1 = squeeze(mat1(j,:,:,:));
        w2 = squeeze(mat2(i,:,:,:,j));
        w(i,:,:,:) = squeeze(w(i,:,:,:)) + convn(w2,w1);
    end
end

end

