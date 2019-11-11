%% This function is to estimate the parameter and latent ratio using expected likelihood and objective function
%%%% Input: 
%        connection_revealed: half-vectorized incomplete network 
%%%% Output:
%        omega_0: estimated latent ratio
%        pi_0: optimized cutoff
%        BB: estimated parameter
%        pihat_mod: fitted probabilities

function [omega_0 pi_0 BB pihat_mod] = ergm_modified(connection_revealed)
shared_neighbor = squareform(double(connection_revealed))*squareform(double(connection_revealed));
num_nodes=size(shared_neighbor,1);
for i=1:num_nodes
shared_neighbor(i,i)=0;
end
shared_neighbor=squareform(full(shared_neighbor))';


degvec = sum(squareform(connection_revealed),1);
degm = degvec'*degvec;
for i=1:num_nodes
degm(i,i)=0;
end
degprod=squareform(full(degm))';


% log-modified regression
QQ_list=zeros(1701,3);
ii=1;
for omega_0=0.01:0.01:0.9
    for pi_0=0.1:0.01:0.3
        BB = mnrfit_modified_simple(double([shared_neighbor degprod]),double(connection_revealed)',omega_0);

        pihat_mod = 1-1./(exp([ones(size(double([shared_neighbor degprod]),1),1) double([shared_neighbor degprod])]*BB)+1);
        logmod_decision = pihat_mod>pi_0;

        % FP
        fp_logmod = sum(logmod_decision(connection_revealed==0)==1);

        % FN
        fn_logmod = sum(logmod_decision(connection_revealed==1)==0);

        
        tp = sum(logmod_decision(connection_revealed==1)==1);
        
        obconn = sum(connection_revealed==1);
        
        Prdconn = sum(logmod_decision);
        
        qqval = tp/obconn*tp/Prdconn;
        
        % quality quantity maximizer tp/observed conns*tp/predicted conns
        QQ_list(ii,1)=pi_0;
        QQ_list(ii,2)=omega_0;
        QQ_list(ii,3)=qqval;
        
        ii=ii+1;
        %ii
    end
end

optimal_QQidx = find(QQ_list(:,3)==max(QQ_list(:,3)));
omega_0 = QQ_list(optimal_QQidx(1),2);
pi_0 =  QQ_list(optimal_QQidx(1),1);

BB = mnrfit_modified_simple(double([shared_neighbor degprod]),double(connection_revealed)',omega_0);
pihat_mod = 1-1./(exp([ones(size(double([shared_neighbor degprod]),1),1) double([shared_neighbor degprod])]*BB)+1);

end