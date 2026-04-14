clc;
clear;
close all;

I = imread("C:\Users\50518\Desktop\sea\pic\多样性数据\018.jpg");%读入原图像
figure,imshow(I);
I = initializeImage(I);
figure,imshow(I);
% pause;

[height,width,channel] = size(I);

H = 210;
H2 = height;

Src = I(H:H2,:,:);
%pause;

BwImg = detect(Src);%调用局部峰值检测法函数

result = uint8(zeros(height,width,1));
result(H:H2,:,:) = BwImg;

figure,imshow(result);
title('结果')