function [index, Z, S, W,obj] = MGAGR(X, alpha, gamma, sigma, group_num, group_label, maxIter)
%Reference to be updated: "Robust Unsupervised Feature Selection via 
%       Multi-Group Adaptive Graph Representation", Mengbo You, 
%       Aihong Yuan, Dongjian He, Xuelong Li, 2020
%
%% input: 
%       dataset X (n times d, n for #samples, d for #features)
%       hyper parameters: alpha, gamma, sigma
%       the number of group: group_num
%       the label of group: group_label
%       the number of iteration: maxIter
%  output:
%       feature selection index: index
%       (for features) self-representation matrix: Z
%       (for samples) global similarity S
%       (for groups) weight matrix: W
% author: u 20200831

[n, d] = size(X);

%% Initialize input parameters
W = ones(group_num, n)./group_num;
Z = rand(d,d);
S = zeros(n,n);
obj = zeros(maxIter, 1);


%% Initialize similarity matrix for each view
S_temp = S;
for v = 1:group_num
    indx = group_label==v;
    data_v = X(:, indx);
    for i = 1:n
        for j = 1:n
            S_temp(i,j) = exp(-(sum((data_v(i, :) - data_v(j, :)).^2))/sigma);
        end
        S_temp(i,:) = S_temp(i,:)./sum(S_temp(i,:));
    end
    S_d.data{v}=(S_temp+S_temp')./2;
end
clear S_temp indx data_v;
for i = 1:n
    for j = 1:n
        S(i,j) = exp(-(sum((X(i, :) - X(j, :)).^2))/sigma);
    end
    S(i,:) = S(i,:)./sum(S(i,:));
end
S=(S+S')./2;


for iter = 1:maxIter
    iter
    %% update W
    disp('update W...');
    for i = 1:n
        A_i = zeros(n, group_num);
        for v = 1:group_num
            A_i(:,v) = S(:,i)-S_d.data{v}(:,i);
        end
        part_bi = A_i'*A_i;
        part_1v = ones(group_num,1);
        temp_inv = part_bi \ part_1v;
        W(:,i) = temp_inv / (part_1v' * temp_inv + 1e-15);
    end
    clear A_i part_bi part_1v;
    
    %% update S
    disp('update S...');
    for i = 1:n
        B_i = zeros(n, group_num);
        for v = 1:group_num
            B_i(:,v) = S_d.data{v}(:,i);
        end
        a_i = zeros(n, 1);
        for p = 1:n
            a_i(p) = norm(X(i,:)*Z-X(p,:)*Z, 'fro')^2;
        end
        part_m = B_i * W(:,i) - 0.25 * alpha * a_i;
        %disp('update psi...');
        psi = zeros(n, 1);
        temp = part_m - ones(n,n) * part_m / n + 0.5 * mean(psi);
        for j = 1:n
            psi(j) = max(-2*temp(j), 0);
        end
        temp = part_m - ones(n,n) * part_m / n + 0.5 * mean(psi);    
        for j = 1:n
            S(i, j) = max(temp(j), 0);
        end
            S(i,:) = S(i,:)./sum(S(i,:));
    end
    S=(S+S')./2;
    clear B_i a_i part_m psi temp;
    
    %% update Z
    disp('update Z...');
    LapMatrix = diag(sum(S, 1)) - S;
    for loop = 1 : maxIter
        temp = 2 * sqrt(sum(Z.^2, 2)) + 1e-15;
        Q = diag(1./temp);
        temp1 = X' * (eye(n) + alpha * LapMatrix) * X + gamma * Q;
        Z = temp1 \ X' * X;
    end
    clear temp Q temp1;
    
    %% calculate objective function value1
    disp('calculate obj-value...');
    
    temp_formulation1 = 0;
    for i =1:n
        temp_S_j = zeros(n,1);
        for v = 1:group_num
            temp_S_i = temp_S_j + W(v,i)*S_d.data{v}(:,i);
        end
        temp_formulation1 = temp_formulation1 + norm(S(:,i)-temp_S_i,'fro')^2;
    end
    temp_formulation2 = norm(X-X*Z,'fro')^2;
    temp_formulation3 = gamma * sum(sqrt(sum(Z.^2, 2)));%L2,1-norm
    temp_formulation4 = alpha * trace(Z'*X'*LapMatrix * X * Z);
    obj(iter) = temp_formulation1 + temp_formulation2 + temp_formulation3 + temp_formulation4;
    clear temp_formulation1 temp_S_j temp_S_i temp_formulation2 temp_formulation3 temp_formulation4;

end
score=sum((Z.*Z),2);
[~,index]=sort(score,'descend');

end
