function rimage = Fengzhi(image,L)
%image：原图像

scale = 15;%某个点需要比较的范围
[height,width,channel] = size(image);

avg = mean(mean(image));
% bd = 2*std2(image)+avg;%峰值下限，像素低于该临界值的不视作为峰值
bd = avg;

l = scale - 1;%需要扩展的像素数
sl = l/2;%单边宽度
st = ceil(scale/2);%起始位置
timage = zeros(height+l,width+l);%用于比较的中间图像
for i = st:st+height-1
    for j = st:st+width-1
        timage(i,j) = image(i-sl,j-sl);
    end
end
timage = uint8(timage);%仅仅是为了便于显示图像
rimage = zeros(height,width);%用来存放结果的图像

%进行峰值检测
counter = 0;%统计峰值个数
for i = st:st+height-1
    for j = st:st+width-1
        if timage(i,j) > bd
            stad = 0;%用于判断几个方向满足条件
            cnt = 0;%记录每个方向满足条件的像素数

            %横向判断
            for k = 1:sl
                if timage(i,j) < timage(i,j+k) | timage(i,j) < timage(i,j-k)
                    break;
                else
                    cnt = cnt + 1;
                end
            end
            if sl == cnt
                stad = stad+1;
            end
            cnt = 0;%计数器清零

            %纵向判断
            for k = 1:sl
                if timage(i,j) < timage(i+k,j) | timage(i,j) < timage(i-k,j)
                    break;
                else
                    cnt = cnt + 1;
                end
            end
            if sl == cnt
                stad = stad+1;
            end
            cnt = 0;%计数器清零

            %正45°斜向判断
            for k = 1:sl
                if timage(i,j) < timage(i-k,j+k) | timage(i,j) < timage(i+k,j-k)
                    break;
                else
                    cnt = cnt + 1;
                end
            end
            if sl == cnt
                stad = stad+1;
            end
            cnt = 0;%计数器清零

            %负45°斜向判断
            for k = 1:sl
                if timage(i,j) < timage(i-k,j-k) | timage(i,j) < timage(i+k,j+k)
                    break;
                else
                    cnt = cnt + 1;
                end
            end
            if sl == cnt
                stad = stad+1;
            end

            if stad == 4%判断几个方向满足条件
                rimage(i-sl,j-sl) = 255;
                counter = counter + 1;
            end
        end
    end
end
rimage = uint8(rimage);
% figure();imshow(rimage);

%统计数组
sta = zeros(1,counter);
k = 1;
for i = 1:1:height
    for j = 1:1:width
        if 255 == rimage(i,j)
            sta(1,k) = image(i,j);
            k = k + 1;
        end
    end
end
u=mean(sta);
thita=std2(sta);
threshold=L*thita+u;
if threshold > 255
    threshold = 255;
end

for i = 1:1:height
    for j = 1:1:width
        if 255 == rimage(i,j)
            if threshold > image(i,j);
                rimage(i,j) = 0;
            end
            k = k + 1;
        end
    end
end
end