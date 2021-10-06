function driver()
close all
fsz = 16; % Fontsize
nt = 5; % trial mesh is nt-by-nt
N = 10; % the number of neurons
tol = 1e-4; % stop if ||J^\top r|| <= tol
iter_max = 5000;  % max number of iterations allowed
[GDf,GDg] = GD(nt,N,tol,iter_max);
[SGf,SGg] = SG(nt,N,tol,iter_max);
[NAGf,NAGg] = NAG(nt,N,tol,iter_max);
[SNAGf,SNAGg] = SNAG(nt,N,tol,iter_max);
[Adamf,Adamg] = Adam(nt,N,tol,iter_max);
[SAdamf,SAdamg] = SAdam(nt,N,tol,iter_max);
%
figure(3);clf;
subplot(2,1,1);
hold on;
plot((1:length(GDf))',GDf,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','GD');

plot((1:length(NAGf))',NAGf,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','NAG');

plot((1:length(Adamf))',Adamf,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','Adam');


legend;
grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
subplot(2,1,2);
hold on;
plot((1:length(GDg))',GDg,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','GD');

plot((1:length(NAGg))',NAGg,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','NAG');

plot((1:length(Adamg))',Adamg,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','Adam');

legend
grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('|| grad f||','Fontsize',fsz);

figure(4);clf;
subplot(2,1,1);
hold on;
plot((1:length(SGf))',SGf,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','SG');
plot((1:length(SNAGf))',SNAGf,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','SNAG');
plot((1:length(SAdamf))',SAdamf,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','SAdam');
legend;
grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
subplot(2,1,2);
hold on;
plot((1:length(SGg))',SGg,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','SG');
plot((1:length(SNAGg))',SNAGg,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','SNAG');
plot((1:length(SAdamg))',SAdamg,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','SAdam');
legend
grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('|| grad f||','Fontsize',fsz);
end