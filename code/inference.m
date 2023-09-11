%% W:the output matrix of L21NPC
%% data:the new data set matrix

distance_matrix=abs([data,ones(length(data),1)]*W);

infer_label=ones(length(data),1);
       
for i=1:length(data)
    infer_label(i)=find(distance_matrix(i,:)==min(distance_matrix(i,:)));
end