% use construction test the accuracy of RBM
function reconstruction()

% reset(gpuDevice(1));
run('setup_paths.m')
% model = load('pretrained_model.mat');
model = load('model30_l2.mat');

% filename1 = './volumetric_data/my_pot/30/train/pot_train_new.mat';
filename2 = './volumetric_data/my_cup/30/train/cup_train.mat';

for goto_l = 2:1:2; %i-th hidden layer
%     reconstruct_one(model.model,filename1,goto_l)
    reconstruct_one(model.model,filename2,goto_l)
end
end

function reconstruct_one(model,filename,goto_l)
run('kernels.m')

data = load(filename);
batch = data.instance;
batch = reshape(batch,[1,size(batch,1),size(batch,2),size(batch,3)]);
% propagate upwards
l = 2;
stride = model.layers{l}.stride;
hidden_presigmoid = myConvolve2(kConv_forward2, batch, model.layers{l}.w, stride, 'forward');
hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
hidden_prob = sigmoid(hidden_presigmoid);
hidden_sample = single(hidden_prob > rand(size(hidden_prob)));
chain = hidden_sample;

while l<goto_l
    % propagate upwards
    l = l+1;
    stride = model.layers{l}.stride;
    hidden_presigmoid = myConvolve(kConv_forward_c, chain, model.layers{l}.w, stride, 'forward');
    hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
    hidden_prob = sigmoid(hidden_presigmoid);
    chain = single(hidden_prob > rand(size(hidden_prob)));
end

while l>2
    % PROPDOWN
    stride = model.layers{l}.stride;
    visible_presigmoid = myConvolve(kConv_backward_c, chain, model.layers{l}.w, stride, 'backward');
    visible_presigmoid = bsxfun(@plus, visible_presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
    chain = sigmoid(visible_presigmoid);
    l = l-1;
end

% PROPDOWN
stride = model.layers{l}.stride;
visible_presigmoid = myConvolve(kConv_backward, chain, model.layers{l}.w, stride, 'backward');
visible_presigmoid = bsxfun(@plus, visible_presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
visible_prob = sigmoid(visible_presigmoid);
% for t=0.1:0.1:0.8
for t=0.5
    show_sample(visible_prob,t)
    view(2)
    
    [~,name,~] = fileparts(filename);
    savefig(strcat(strcat(name,'_l',num2str(goto_l)),'_',num2str(t),'.fig'));
    %     close
end
end


function [y] = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end