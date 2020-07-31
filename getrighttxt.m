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
    %��ȡÿһ��
    a=allclasses_names(i);
    %�ַ���ƴ��
    name=strcat(int2str(i),'.',a{1,1}(1,:));
    
    %��������txt�����Ƿ����������ȵģ�������ڣ���ô����һ�������������飬��¼���Եı��
    for j=1:length(testclasses_name)
        %���е��ַ���Ϊa{1,1}(1,:)
        b=testclasses_name(j);
        c=b{1,1}(1,:);
        if strcmp(a,c)==1
            %ÿ�ζ���72��Ԫ��
            test_name=strcat(int2str(i),'.',c);
            test_class=[test_class,i];
            testclasses_name_num=[testclasses_name_num cellstr(test_name)];
            flag=0;
        end    
    end
    %����׷��Ԫ��
    name;
    if flag==1
        trainclasses_name_num=[trainclasses_name_num cellstr(name)];
    end
    allclasses_name_num=[allclasses_name_num cellstr(name)];
    flag=1;
end
%����ת��
allclasses_name_num=allclasses_name_num';
testclasses_name_num=testclasses_name_num';
trainclasses_name_num=trainclasses_name_num';
%������test�ı��е������滻Ϊ���+���֣���Ӧ�ģ�
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
%�������ѵ�����У�
[nrows,ncols]= size(trainclasses_name_num);
trainclasses_filename = 'train_classes.txt';
fid = fopen(trainclasses_filename, 'w');
for row=1:nrows
    fprintf(fid, '%s\n', trainclasses_name_num{row,:});
end
fclose(fid);
