function [result,ContrastArray] = ContrastAnalysis(src,bsrc,tgwidth,tgheight,thita)
%检测每个峰的局部对比度，并对所有峰的局部对比度进行分析判断
%src：原图像；bsrc：峰值检测结果图像；tgwidth：目标的宽度；tgheight：目标的高度
%thita：根据对比度信息二值化的阈值计算系数

    result = bsrc;%存放检测结果的图像
    [height,width,nchannels] = size(src);
    ContrastTable = ContrastInfoCalc(src,bsrc,tgwidth,tgheight);
    temp = ContrastTable(:,:,1);
    pos = find(temp~=0);
    tarray = temp(pos);
    avg = mean(tarray);
    sd = mean(abs(tarray-avg));
%     sd = mean(abs(ContrastTable(pos)-avg));
    for i = 1:height
        for j = 1:width
            if bsrc(i,j) >= 240
                frontcnt = ContrastTable(i,j,2);
                bkcnt = ContrastTable(i,j,3);
                if double(frontcnt/(frontcnt+bkcnt)) > 0.3%此处的比例值有待调整
                    shift = 1;
                else
                    shift = 8;
                end
                if ContrastTable(i,j,1)>avg+thita*sd%若对比度高于除零后的平均对比度，则峰值面积扩展，否则清零
                    for k=i-ContrastTable(i,j,4):i+ContrastTable(i,j,5)
                        for l=j-ContrastTable(i,j,6):j+ContrastTable(i,j,7)
                            if src(k,l)>=ContrastTable(i,j,8)-shift
                                result(k,l)=255;
                            end
                        end
                    end
                else
                    result(i,j)=0;
                end
            end
        end
    end
end