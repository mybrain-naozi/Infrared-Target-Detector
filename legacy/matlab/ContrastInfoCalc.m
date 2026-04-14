function ContrastTable = ContrastInfoCalc(src,bsrc,tgwidth,tgheight)
%ContrastTable是一个存放各点对比度信息的多通道矩阵，1~8通道依次为：
%对比度/前景数目/背景数目/上/下/左/右/局部灰度最大值

    [height,width,nchannels] = size(src);
    ContrastTable = zeros(height,width,8);
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
                            frontcnt=frontcnt+double(1);%前景数
                        else
                            bk=bk+double(src(k,l));
                            bkcnt=bkcnt+double(1);%背景数
                        end
                    end
                end
                ContrastTable(i,j,4)=up;
                ContrastTable(i,j,5)=down;
                ContrastTable(i,j,6)=left;
                ContrastTable(i,j,7)=right;                
                ContrastTable(i,j,8)=mx;
                if 0 == bkcnt
                    ContrastTable(i,j,1)=0.000;
                else
                    ContrastTable(i,j,1)=(front/frontcnt)/(bk/bkcnt);
                    ContrastTable(i,j,2)=frontcnt;
                    ContrastTable(i,j,3)=bkcnt;
                end
            end
        end
    end
    
end