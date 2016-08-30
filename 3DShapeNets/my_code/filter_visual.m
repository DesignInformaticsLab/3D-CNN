% function filter_visual()
% load('./pretrained_model.mat');
% w = rec_conv(model,3); % filter of which rbm
% save('w_projected.mat','w');
% % plot_filter(w)
% for i=1:10:16
%     figure;
%     cnt = 1;
%     for t=6:-1:1
%         subplot(2,3,cnt);
%         show_sample(squeeze(w(i,:,:,:)),t)
%         title(strcat('threshold ',num2str(t)))
%         cnt = cnt + 1;
%     end
%
%
% end
% end

function filter_visual()
load('model30_l2.mat');
w = (rec_conv(model,3)); % filter of which rbm

save('w_projected.mat','w');
% plot_filter(w)
a = max(max(max(max(w))));
b = min(min(min(min(w))));
w = (w-b)/(a-b);
tmp = sort(reshape(w,[],1),'descend');
% t = tmp(ceil(length(tmp)/10))
t = 0.1
figure;
cnt = 1;
for i=1:2:16
    subplot(2,4,cnt);
    show_sample(squeeze(w(i,:,:,:)),t)
    title(strcat('filter ',num2str(i)))
    cnt = cnt + 1;
    
end
figure;
cnt = 1;
for i=2:2:16
    subplot(2,4,cnt);
    show_sample(squeeze(w(i,:,:,:)),t)
    title(strcat('filter ',num2str(i)))
    cnt = cnt + 1;
    
end
% figure;
% cnt = 1;
% for i=3:4:32
%     subplot(2,4,cnt);
%     show_sample(squeeze(w(i,:,:,:)),t)
%     title(strcat('filter ',num2str(i)))
%     cnt = cnt + 1;
%     
% end
% figure;
% cnt = 1;
% for i=4:4:32
%     subplot(2,4,cnt);
%     show_sample(squeeze(w(i,:,:,:)),t)
%     title(strcat('filter ',num2str(i)))
%     cnt = cnt + 1;
%     
% end
end


function plot_filter(w)

figure;
assert(size(w,1)>=4);
for i=1:4
    subplot(2,2,i)
    vol3d('cdata', squeeze(w(i,:,:,:)), 'xdata', [0 1], 'ydata', [0 1], 'zdata', [0 1]);
    colormap(bone(256));
    alphamap([0 linspace(0.1, 0, 2)]);
    %     axis([0.1,0.9,0.1,0.9,0.1,0.9])
    set(gcf, 'color', 'w');
    view(3);
end

end


function mat2 = rec_conv(model,layer_idx)
kernels
if layer_idx==1
    mat2 = (model.layers{2}.w);
    return
end
mat2 = (model.layers{layer_idx+1}.w); % upper layer
while layer_idx~=1
    mat1 = (model.layers{layer_idx}.w); % lower layer
    stride = model.layers{layer_idx}.stride;
%     b = model.layers{layer_idx}.b;
    if(layer_idx==2)
        for i=1:1:size(mat2,1)
            mat2_tmp(i,:,:,:) = squeeze(myConvolve2(kConv_backward, mat2(i,:,:,:,:), mat1, stride, 'backward'));           
        end
        mat2 = mat2_tmp;
%         mat2 = bsxfun(@plus, mat2, permute(b, [5,1,2,3,4]));
    else
        mat2 = myConvolve(kConv_backward_c, mat2, mat1, stride, 'backward');
%         mat2 = bsxfun(@plus, mat2, permute(b, [5,1,2,3,4]));
    end
    layer_idx = layer_idx - 1;
end
end

