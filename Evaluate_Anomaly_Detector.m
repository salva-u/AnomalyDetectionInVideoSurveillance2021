clc
clear all
close all
 

C3D_CNN_Path='./Dataset/TestingVMZFeatures/'; % features for videos
Testing_VideoPath='./Dataset/TestingVideos/'; % Path of mp4 videos
AllAnn_Path='./Temporal_Anomaly_Annotation.txt'; % Path of Temporal Annotations
Model_Score_Folder='./EvalRes_VMZ/';  % Path of Pretrained Model score on Testing videos (32 numbers for 32 temporal segments)
Paper_Results='./';   % Path to save results.
 
All_Videos_scores=dir(Model_Score_Folder);
All_Videos_scores=All_Videos_scores(3:end);
nVideos=length(All_Videos_scores);
frm_counter=1;
All_Detect=zeros(1,1000000);
All_GT=zeros(1,1000000);
Ann = readtable(AllAnn_Path);
% Ann.Var1{1} = 'Abuse028_x264.mp4';
% Ann.Var2{1} = 'Abuse';
% Ann.Var3(1) = 165;
% Ann.Var4(1) = 240;
% Ann.Var5(1) = -1;
% Ann.Var6(1) = -1;
for ivideo=1:nVideos
    ivideo

    name = mat2str(cell2mat(Ann.Var1(ivideo)));
    name = name(2:end-1)
    All_Videos_scores(ivideo).name(1:end-4)
    check=strmatch(All_Videos_scores(ivideo).name(1:end-3),name(1:end-3));
    if isempty(check)
         error('????') 
    end
   

    VideoPath=[Testing_VideoPath,'/', All_Videos_scores(ivideo).name(1:end-4),'.mp4'];
    ScorePath=[Model_Score_Folder,'/', All_Videos_scores(ivideo).name(1:end-4),'.mat'];
    disp("working")

  %% Load Video
    try
        xyloObj = VideoReader(VideoPath);
    catch

       error('???')
    end

    Predic_scores=load(ScorePath, '-ASCII'); %maybe load differently 
    fps=30;
    Actual_frames=round(xyloObj.Duration*fps);
%------------ fine till here ---------
    Folder_Path=[C3D_CNN_Path,'/',All_Videos_scores(ivideo).name(1:end-4)];
    AllFiles=dir([Folder_Path,'.txt']);
    nFileNumbers=32;
    nFrames_C3D=89*16;  % As the features were computed for every 16 frames


%% 32 Shots
    %Detection_score_32shots=zeros(1,Actual_frames);
    Thirty2_shots= round(linspace(1,Actual_frames,33));
    Final_score=[];
    p_c=0;

    for ishots=1:length(Thirty2_shots)-1

        p_c=p_c+1;
        ss=Thirty2_shots(ishots);
        ee=Thirty2_shots(ishots+1);
        ff = int8(ee)-int8(ss);
%         if ishots==length(Thirty2_shots)
%             ee=Thirty2_shots(ishots+1);
%         end
% 
%         if ee<ss
%             Detection_score_32shots((ss-1)+1:(ss-1)+1+15)=Predic_scores(p_c);   
%         else
%             Detection_score_32shots((ss-1)*16+1:(ee-1)*16+16)=Predic_scores(p_c);
%         end
        score = repmat(Predic_scores(p_c),1,ff);
        if p_c == 1
           Final_score = score;
        else
            Final_score = horzcat(Final_score, score);
        end

    end


    %Final_score=  [Detection_score_32shots,repmat(Detection_score_32shots(end),[1,Actual_frames-length(Detection_score_32shots)])];
    GT=zeros(1,Actual_frames);

    %for ik=1:size(Testing_Videos1.Ann,1)
            startFR = Ann.Var3(ivideo);
            endFR = Ann.Var4(ivideo);
            %st_fr=max(Testing_Videos1.Ann(ik,1),1); 
            %end_fr=min(Testing_Videos1.Ann(ik,2),Actual_frames);
           
   % end


    if startFR==-1 && endFR==-1   % For Normal Videos
        GT=zeros(1,Actual_frames);
    else 
         GT(startFR:endFR)=1;
    end


    % Final_score= ones(1,p_c);%32 bags 
     %subplot(2,1,1); bar(Final_score)
    % subplot(2,1,2); bar(GT)

    All_Detect(frm_counter:frm_counter+length(Final_score)-1)=Final_score;
    All_GT(frm_counter:frm_counter+Actual_frames -1)= GT;
    frm_counter=frm_counter+length(Final_score);


end


All_Detect=(All_Detect(1:frm_counter-1));
All_GT=All_GT(1:frm_counter-1);
scores=All_Detect;
[so,si] = sort(scores,'descend');
tp=All_GT(si)>0;
fp=All_GT(si)==0;
tp=cumsum(tp);
fp=cumsum(fp);
nrpos=sum(All_GT);
rec=tp/nrpos;
fpr=fp/sum(All_GT==0);
prec=tp./(fp+tp);
AUC1 = trapz(fpr ,rec );
% You can also use the following codes
[X,Y,T,AUC] = perfcurve(All_GT,All_Detect,1);

 
