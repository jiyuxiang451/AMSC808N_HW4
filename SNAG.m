function [fall,norg] = SNAG(nt,N,tol,iter_max)
fsz = 16; % fontsize
%% setup training mesh
t = linspace(0,1,nt+2);
[xm,ym] = meshgrid(t,t);
I = 2:(nt+1);
xaux = xm(I,I);
yaux = ym(I,I);
xy = [xaux(:),yaux(:)]';
Ntrain = nt*nt; % the number of training points
%% initial guess for parameters
N = 10; % the number of hidden nodes (neurons)
npar = 4*N; % the total number of parameters
w = ones(npar,1); % initially all parameters are set to 1
temp1 = ones(npar, 1); % initially set temp1 
temp2 = ones(npar, 1); % initially set temp2
%% random choice of indices
% Define size of batch for random choice of indices
batch_size = 5;
batch_ind = zeros(batch_size);
%%
[r,J] = Res_and_Jac(w,xy);
f = F(r);
g = J'*r/Ntrain;
nor = norm(g);
fprintf('Initially: f = %d, nor(g) = %d\n',f,nor); 
%%
tic % start measuring the CPU time
iter = 2;
norg = zeros(iter_max+1,0);
fall = zeros(iter_max+1,0);
muall = zeros(iter_max+1, 0); % set mu_k for each step
ytemp = ones(npar, 1); % set y_i for each step
norg(1) = nor;
fall(1) = f;
% TODO: DEFINE STEPSIZE 
alpha = 0.25;
% above is learning rate that I chose
while nor > tol && iter < iter_max
    muall(iter) = 1 - 3 / (4+iter);
    ytemp = (1+muall(iter)) * temp1 - muall(iter) * temp2;
    [r,J] = Res_and_Jac(ytemp,xy);
    % batch
    for i = 1:batch_size
            batch_ind(i) = randi(Ntrain);
    end
    g = 0;
    for i = 1:batch_size
            g = g + (J(batch_ind(i), :)' * r(batch_ind(i)));
    end
    g = g / batch_size;
    w = ytemp - alpha * g;
    [r,J] = Res_and_Jac(w,xy);
    f = F(r);
    g = J'*r/Ntrain;
    nor = norm(g);     
    fprintf('iter %d: f = %d, norg = %d\n',iter,f,nor);
    norg(iter) = nor;
    fall(iter) = f;
    temp2 = temp1;
    temp1 = w;
    iter = iter + 1;
end
fprintf('iter # %d: f = %.14f, |df| = %.4e\n',iter-1,f,nor);
cputime = toc;
fprintf('CPUtime = %d, iter = %d\n',cputime,iter);
%% visualize the solution
nt = 101;
t = linspace(0,1,nt);
[xm,ym] = meshgrid(t,t);
[fun,~,~,~] = ActivationFun();
[v,W,u] = param(w);
[f0,f1,g0,g1,~,~,~,~,h,~,~,~,exact_sol] = setup();
A = @(x,y)(1-x).*f0(y) + x.*f1(y) + (1-y).*(g0(x)-((1-x)*f0(0)+x*f1(0))) + ...
     y.*(g1(x)-((1-x)*f0(1)+x*f1(1)));
B = h(xm).*h(ym);
NNfun = zeros(nt);
for i = 1 : nt
    for j = 1 : nt
        x = [xm(i,j);ym(i,j)];
        NNfun(i,j) = v'*fun(W*x + u);
    end
end
sol = A(xm,ym) + B.*NNfun;
esol = exact_sol(xm,ym);
err = sol - esol;
fprintf('max|err| = %d, L2 err = %d\n',max(max(abs(err))),norm(err(:)));

%
figure(1);clf;
contourf(t,t,sol,linspace(min(min(sol)),max(max(sol)),20));
colorbar;
set(gca,'Fontsize',fsz);
xlabel('x','Fontsize',fsz);
ylabel('y','Fontsize',fsz);

%
figure(2);clf;
contourf(t,t,err,linspace(min(min(err)),max(max(err)),20));
colorbar;
set(gca,'Fontsize',fsz);
xlabel('x','Fontsize',fsz);
ylabel('y','Fontsize',fsz);
%
figure(3);clf;
subplot(2,1,1);
fall(iter:end) = [];
plot((1:iter-1)',fall,'Linewidth',2,'Marker','.','Markersize',20);
grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
subplot(2,1,2);
norg(iter:end) = [];
plot((1:iter-1)',norg,'Linewidth',2,'Marker','.','Markersize',20);
grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('|| grad f||','Fontsize',fsz);
end

%% the objective function
function f = F(r)
    f = 0.5*r'*r/length(r);
end
