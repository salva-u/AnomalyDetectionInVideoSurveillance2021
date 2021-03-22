video = VideoReader('./C3D/input/Arrest002_x264.mp4');
frame = read(video, 300);
img = imread('./Paper_Results/FinalROC_Comparison.jpg');
imshow(frame);
%connvert to grayscale 
img_gray = rgb2gray(frame);
max(img_gray);

%convert to binary 
img_bin = imbinarize(img_gray);
imshow(img_bin)
figure
subplot(1,3,1)

%getting the different intensities
img_r = frame(:,:,1); %extraction of red matrix
img_g = frame(:,:,2);%extraction of green matrix
img_b = frame(:,:,3); %extraction of blue matrix

%histogram equalization to enahance image; increase the contrast
enhanced_frame = histeq(img_gray);
imhist(enhanced_frame)
imhist(img_gray)
imshow(enhanced_frame)

%Applying Image Detection 
frame_edges = edge(enhanced_frame, 'log');
imshow(frame_edges)


