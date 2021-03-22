clc %this will clear the console 
clear all % this will remove all variables 
close all
% each variable is a type of matrix 

ROC_PathAL='./Paper_Results';
All_files=dir([ROC_PathAL,'/*.mat']);
%All_files=All_files(3:end);
Colors={'b','c','k','r','y'};

AUC_All=[];
k = 0;
for i=1:length(All_files) 
     
    
    FilePath=[ROC_PathAL,'/',All_files(i).name]
    load(FilePath)
    
    plot(X,Y,'Color',Colors{i},'LineWidth',3.5);
    hold on;
    AUC_All=[AUC_All;AUC] % the semi colon is used as a seperator to add a new row
    clear X  Y
    
end
% [ 9 9 0; 0 9 9; 9 0 9] 3x3 matrix
AUC_All*100

legend({'Binary classifier','Lu et al.','Hassan et al.','Model''s result','Own Trained Result'},'FontSize',16,'Location','southeast');
xlabel('False Positive Rate','FontWeight','normal','FontSize',18);
ylabel('True Positive Rate','FontWeight','normal','FontSize',18);
set(gca,'FontWeight','normal','FontSize',12);

grid on

%hello(row, col)

 
