function [bestfit,bestpos,cg]=ICPO(Pop_size,Tmax,lb,ub,dim,fobj)

    N1=Pop_size; %% Is the initial population size.
    N_min=round(0.8*Pop_size); %% Is the minimum population size.
    A=2; %% The number of cycles
    Tf=0.8; %% The percentage of the tradeoff between the third and fourth defense mechanisms
    po=initialization(Pop_size,dim,ub,lb); % Initialize the positions of crested porcupines

    for i=1:Pop_size
        fit(i)=fobj(po(i,:));
    end
    % Update the best-so-far solution
    [bestfit,index]=min(fit);
    bestpos=po(index,:);    
    X=po;
    
    %%  Optimization Process of CPO
    for t = 1:Tmax 
        r2=rand;
        for i=1:Pop_size
    
            U1=rand(1,dim)>rand;
            if rand<rand %% Exploration phase
                if rand<rand %% First defense mechanism 
                    po(i,:)=po(i,:) + abs(bestpos-po(i,:)).*rand;
                else %% Second defense mechanism
                    y=(po(i,:)+po(randi(Pop_size),:))/2;
                    po(i,:)=(U1).*po(i,:)+(1-U1).*(y+rand*(po(randi(Pop_size),:)-po(randi(Pop_size),:)));
                end
            else
                Yt=2*rand*(1-t/(Tmax))^(t/(Tmax));
                U2=rand(1,dim)<0.5*2-1;
                S=rand*U2;
                if rand<Tf %% Third defense mechanism
                    %% 
                    St=exp(fit(i)/(sum(fit)+eps)); % plus eps to avoid division by zero
                    S=S.*Yt.*St;
                    po(i,:)= (1-U1).*po(i,:)+U1.*(po(randi(Pop_size),:)+St*(po(randi(Pop_size),:)-po(randi(Pop_size),:))-S); 
    
                else %% Fourth defense mechanism
                    po(i,:)= (bestpos+(rand*(1-r2)+r2)*(bestpos-po(i,:))); 
                end
            end

            %% Return the search agents that exceed the search space's bounds
            for j=1:size(po,2)
                if  po(i,j)>ub(j)
                    po(i,j)=lb(j)+rand*(ub(j)-lb(j));
                elseif  po(i,j)<lb(j)
                    po(i,j)=lb(j)+rand*(ub(j)-lb(j));
                end
            end  

            nfit=fobj(po(i,:));
            %% update Global & Personal best solution
            if  fit(i)<nfit
                po(i,:)=X(i,:);    % Update local best solution
            else
                X(i,:)=po(i,:);
                fit(i)=nfit;
                if  fit(i)<=bestfit
                    bestpos=po(i,:);    % Update global best solution
                    bestfit=fit(i);
                end
            end
    
        end
%         N=fix(N_min+(N1-N_min)*(1-(rem(t,T/A)/T/A)));
        cg(t)=bestfit;
     end
end


% This function initialize the first population of search agents
function Positions=initialization(SearchAgents_no,dim,ub,lb)

    Boundary_no= size(ub,2); % number of boundaries
    
    % If the boundaries of all variables are equal and user enter a signle
    % number for both ub and lb
    if Boundary_no==1
        Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
    end
    
    % If each variable has a different lb and ub
    if Boundary_no>1
        for i=1:dim
            ub_i=ub(i);
            lb_i=lb(i);
            Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
        end
    end
end
