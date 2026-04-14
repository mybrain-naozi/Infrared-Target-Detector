function result = detect(img)

image = img;
result = img;
[height,width,channel] = size(image);
H=40;%划分方法有待进一步优化
V=width/1;
fenshuH = floor(height/H);%垂直方向划分的份数
fenshuL = floor(width/V);%水平方向划分的份数
rimage = zeros(height,width);
L=1.76;
tgwidth=15;
tgheight=15;%搜索范围有待优化
thita=2;%用于对比度阈值函数

image=AvgAdjust(image,height,width);%图像的平均灰度校正
% figure,imshow(image);完事删掉
% pause;

for i = 1:fenshuH
    for j = 1:fenshuL
        if i == fenshuH
            if j == fenshuL
                timg = image((fenshuH-1)*H+1:height,(fenshuL-1)*V+1:width);
                ftimg = Fengzhi(timg,L);
                result((fenshuH-1)*H+1:height,(fenshuL-1)*V+1:width) = ContrastAnalysis(timg,ftimg,tgwidth,tgheight,thita);
            else
                timg = image((fenshuH-1)*H+1:height,(j-1)*V+1:j*V);
                ftimg = Fengzhi(timg,L);
                result((fenshuH-1)*H+1:height,(j-1)*V+1:j*V) = ContrastAnalysis(timg,ftimg,tgwidth,tgheight,thita);
            end
        else
            if j == fenshuL
                timg = image((i-1)*H+1:i*H,(fenshuL-1)*V+1:width);
%                 figure,imshow(timg);
%                 pause;
                ftimg = Fengzhi(timg,L);
%                 figure,imshow(ftimg);
%                 pause;                
                result((i-1)*H+1:i*H,(fenshuL-1)*V+1:width) = ContrastAnalysis(timg,ftimg,tgwidth,tgheight,thita);
%                 figure,imshow(result((i-1)*H+1:i*H,(fenshuL-1)*V+1:width));
%                 pause;
            else
                timg = image((i-1)*H+1:i*H,(j-1)*V+1:j*V);
                ftimg = Fengzhi(timg,L);
                result((i-1)*H+1:i*H,(j-1)*V+1:j*V) = ContrastAnalysis(timg,ftimg,tgwidth,tgheight,thita);
            end
        end
    end
end
end