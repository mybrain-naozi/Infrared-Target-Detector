function [avg,sd] = ContrastAvg(src,bsrc,tgwidth,tgheight,thita)
%计算全局对比度的平均值（不包括对比度为零的峰）

    [height,width,nchannels] = size(src);
    ContrastArray=[];%存放对比度的数组
    concnt=0;%对比度计数器
    zcnt=0;%对比度为零的计数器
    for i = 1:height
        for j = 1:width
            if bsrc(i,j) >= 240
                up = tgheight;%向上扩展的像素数
                down = up;
                left = 2*tgwidth;%向左扩展的像素数
                right = left;
                %防止搜索越界
                if i-up <= 0
                    up = i-1;
                end
                if i+down > height
                    down = height - i;
                end
                if j-left <= 0
                    left = j-1;
                end
                if j+right > width
                    right = width - j;
                end    
                front=0;
                frontcnt=0;
                bk=0;
                bkcnt=0;
                %寻找局部落差
                localArray=src(i-up:i+down,j-left:j+right);%局部图像
                mx=max(max(localArray));
                mn=mean(mean(localArray));
                maxDiff=mx-mn;%局部落差
                threshold=mx-maxDiff*0.9;
                for k=i-up:i+down
                    for l=j-left:j+right
                        if src(k,l)>=threshold
                            front=front+double(src(k,l));
                            frontcnt=frontcnt+double(1);
                        else
                            bk=bk+double(src(k,l));
                            bkcnt=bkcnt+double(1);
                        end
                    end
                end
                concnt=concnt+1;
                if 0 == bkcnt
                    zcnt=zcnt+1;
                    ContrastArray(1,concnt)=0.000;
                else
                    ContrastArray(1,concnt)=(front/frontcnt)/(bk/bkcnt);
                end
            end
        end
    end
    pos = find(ContrastArray~=0);
    tempArray =  ContrastArray(pos);
    avg = mean(tempArray);
    sd = mean(abs(tempArray-avg));
end