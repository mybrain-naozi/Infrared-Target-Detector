function ri=AvgAdjust(image,height,width)
%图像平均值校正

    ri = image;

    avgPerLine = zeros(1,height);
    for i = 1:height
        temp = image(i,:);
        avgPerLine(1,i) = uint8(mean(temp));
    end
    Max = uint8(max(avgPerLine));

    for i = 1:height
        ri(i,:) = image(i,:)+(Max-avgPerLine(1,i));
    end
end