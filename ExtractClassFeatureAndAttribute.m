clear
%%%%%%%%%%%%%%%%%%%%%获取训练类与测试类的2048维视觉特征与312维语义特征%%%%%%%%%%%%%%%%%%%%%
load('trainANDtestClass.mat')
load('res101.mat')
load('att_splits.mat')
pict_len=length(labels);
clear image_files
trainclasses_features=[];
testclasses_features=[];
trainclasses_attr=[];
testclasses_attr=[];
trainclasses_labels=[];
j=1;
k=1;
l=1;
m=1;
n=1;
o=1;
for i=1:pict_len
    if ismember(labels(i),trainclasses)
        trainclasses_features(:,j)=features(:,i);
        trainclasses_attr(:,l)=att(:,labels(i));
        trainclass_labels(n,:)=labels(i);
        l=l+1;
        j=j+1;
        n=n+1;
    end
    if ismember(labels(i),testclasses)
        testclasses_features(:,k)=features(:,i);
        testclasses_attr(:,m)=att(:,labels(i));
        testclass_labels(o,:)=labels(i);
        k=k+1;
        m=m+1;
        o=o+1;
    end
end
testclasses_features=testclasses_features';
trainclasses_features=trainclasses_features';
trainclasses_attr=trainclasses_attr';
testclasses_attr=testclasses_attr';
clear i j k l m n o allclass val_loc trainval_loc train_loc 
clear allclass pict_len test_unseen_loc test_seen_loc original_att
save('AWA2.mat')