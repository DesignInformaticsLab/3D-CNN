kernels;

% filename2 = './my_code/more/cup_train_sal.mat';
% filename2 = './volumetric_data/chair/30/train/chair_000000415_12.mat';
filename2 = './volumetric_data/my_cup_sal/30/train/cup_train_saliency.mat';
load('model30_l2.mat');
w = model.layers{2}.w; % filter of which rbm


data = load(filename2);
batch = data.instance;
batch = reshape(batch,[1,size(batch,1),size(batch,2),size(batch,3)]);
for fi = 1:1:16;
    
    l=2;
    figure;    
    
    stride = model.layers{l}.stride;
    hidden_presigmoid = myConvolve2(kConv_forward2, batch, model.layers{l}.w, stride, 'forward');
    hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
    hidden_prob = sigmoid(hidden_presigmoid);
    % hidden_sample = single(hidden_prob > rand(size(hidden_prob)));
    hidden_sample = single(hidden_prob > 0.5);
    chain = hidden_sample;
    
    
    ww =  model.layers{l}.w;
    test_ = fi;
    if test_
        num_channel = size(model.layers{l}.w,1);
        for i=1:num_channel
            if i~=test_
                %         chain(1,:,:,:,i) = 0;
                ww(i,:,:,:)  = 0;
            else
                ww(i,:,:,:) = num_channel * ww(i,:,:,:);
            end
        end
    end
    
    % PROPDOWN
    stride = model.layers{l}.stride;
    % it seems there is no difference between myConvolve2 and myConvolve
    visible_presigmoid = myConvolve2(kConv_backward, chain, ww, stride, 'backward');

    th = 0.2;
    subplot(3,2,1);t=squeeze(w(fi,:,:,:));show_slice(t);title('w');
    subplot(3,2,2);t=squeeze(chain(1,:,:,:,fi));show_slice(t); title('h');
    subplot(3,2,3);t=squeeze(visible_presigmoid(1,:,:,:));show_slice(t);title('w''h');
    
    visible_presigmoid = bsxfun(@plus, visible_presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
    subplot(3,2,4);t=model.layers{l}.b;show_slice(t);title('b');
    subplot(3,2,5);t=squeeze(visible_presigmoid(1,:,:,:));show_slice(t);title('w''h+b');

    visible_prob = sigmoid(visible_presigmoid);
    subplot(3,2,6);t=squeeze(visible_prob (1,:,:,:));show_slice(t);title('v');
end


