clear;
%%%%%%%%%%%读取类别文件，获取所有类别的标号%%%%%%%%%%%
allclasses_filename='allclasses.txt';
allclasses_names=textread(allclasses_filename,'%s');
allclasses=[];
for i=1:50
    allclasses_names(i);
    a= strsplit(string(allclasses_names(i)),'.');
    allclasses(end+1)=a(1);
    i=i+1;
end
allclasses=allclasses';

%%%%%%%%%%%读取类别文件，获取测试类别的标号%%%%%%%%%%%
testclasses_filename='test_classes.txt';
testclasses_names=textread(testclasses_filename,'%s');
testclasses=[];
for i=1:10
    testclasses_names(i);
    a= strsplit(string(testclasses_names(i)),'.');
    testclasses(end+1)=a(1);
    i=i+1;
end
testclasses=testclasses';
%%%%%%%%%%%获取训练类别的标号%%%%%%%%%%%
clear a ans i allclasses_filename testclasses_filename testclasses_names allclasses_names;
l_all=length(allclasses);
l_test=length(testclasses);
trainclasses=[];
k=1;
for i=1:l_all
    for j=1:l_test
        if allclasses(i)==testclasses(j)
            break;
        end
        if j==l_test
            trainclasses(end+1)=allclasses(i);
        end
    end
end
trainclasses=trainclasses';
clear i j k ans l_all l_test
save('trainANDtestClass.mat')