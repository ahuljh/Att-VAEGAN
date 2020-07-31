clear all,clc
load('att_splits.mat')
allclasses_name_num=[];
testclasses_filename='testclasses.txt';
testclasses_name=textread(testclasses_filename,'%s');
test_class=[];
testclasses_name_num=[];
train_class=[];
trainclasses_name_num=[];
flag=1;
for i=1:length(allclasses_names)
    %读取每一行
    a=allclasses_names(i);
    %字符串拼接
    name=strcat(int2str(i),'.',a{1,1}(1,:));
    
    %遍历测试txt，看是否存在与其相等的，如果存在，那么建立一个测试类别的数组，记录测试的标号
    for j=1:length(testclasses_name)
        %现有的字符串为a{1,1}(1,:)
        b=testclasses_name(j);
        c=b{1,1}(1,:);
        if strcmp(a,c)==1
            %每次都有72个元素
            test_name=strcat(int2str(i),'.',c);
            test_class=[test_class,i];
            testclasses_name_num=[testclasses_name_num cellstr(test_name)];
            flag=0;
        end    
    end
    %数组追加元素
    name;
    if flag==1
        trainclasses_name_num=[trainclasses_name_num cellstr(name)];
    end
    allclasses_name_num=[allclasses_name_num cellstr(name)];
    flag=1;
end
%数组转置
allclasses_name_num=allclasses_name_num';
testclasses_name_num=testclasses_name_num';
trainclasses_name_num=trainclasses_name_num';
%将所有test文本中的名字替换为序号+名字（对应的）
[nrows,ncols]= size(allclasses_name_num);
allclasses_filename = 'allclasses.txt';
fid = fopen(allclasses_filename, 'w');
for row=1:nrows
    fprintf(fid, '%s\n', allclasses_name_num{row,:});
end
fclose(fid);

[nrows,ncols]= size(testclasses_name_num);
allclasses_filename = 'test_classes.txt';
fid = fopen(allclasses_filename, 'w');
for row=1:nrows
    fprintf(fid, '%s\n', testclasses_name_num{row,:});
end
fclose(fid);
%怎样获得训练的行？
[nrows,ncols]= size(trainclasses_name_num);
trainclasses_filename = 'train_classes.txt';
fid = fopen(trainclasses_filename, 'w');
for row=1:nrows
    fprintf(fid, '%s\n', trainclasses_name_num{row,:});
end
fclose(fid);
