function Img = initializeImage(I)
%该函数将输入图像格式统一调整为无符号8位，单通道图像，分辨率为[640，512]
    if 3 == ndims(I)%通道转为单通道
        I = rgb2gray(I);
    end
    if [512,640]~=size(I)%尺寸转换
        I = imresize(I,[512,640],'bicubic');
    end
    if ~strcmp('uint8',class(I))%格式调整
        I = im2uint8(I);
    end
    timg = I(4 : 509,4 : 637);%亮边抠出后的图像
    Img = imresize(timg,[512,640],'bicubic');
end