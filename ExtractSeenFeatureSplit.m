clear
%%%%%%%%%%%%%%%%%%%%%��ȡѵ������������2048ά�Ӿ�������312ά��������%%%%%%%%%%%%%%%%%%%%%
load('AWA2.mat')
seen_class=trainclasses;
seen_class_labels=trainclass_labels;
seen_class_len=length(seen_class);
seen_class_labels_len=length(seen_class_labels);
for i=1:seen_class_len
    seen_class_set(seen_class(i))=0;
    for j=1:seen_class_labels_len
        if seen_class_labels(j)==seen_class(i)
            seen_class_set(seen_class(i))=seen_class_set(seen_class(i))+1;
        end
    end
end
allclass_len=length(allclasses);
seen_class_set(2,:)=0;
seen_class_set(3,:)=0;
%%ֻ��AWA1��-1�����������ݲ���-1�����һ������ǲ����ࣩ
for k=1:allclass_len-1
    seen_class_set(2,k)=seen_class_set(1,k)/11;
    seen_class_set(3,k)=k;
end
seen_class_feature=trainclasses_features;
%%%%����3*49��set���飬�����ڶ���ȡ����ת��Ϊint��Ϊ0�Ļ�����
%%%%����features��labels��������ڸ���Ļ���ȡ��Ӧset�еڶ��и��������µ�seen_test�У�ʣ�µĴ���seen_train��
%%%%�½�seen_test_labels��seen_train_labels����������Ӧ
%%%%�����õ�4��seen_test��seen_train��seen_test_labels��seen_train_labels�ļ������µ�mat�ļ�
%%%%Python�ֱ���룬ѵ����δ�������ɵ�ƴ�ӣ�����ѵ����Ԥ��
see_train_feature=[];
seen_train_labels=[];
see_test_feature=[];
seen_test_labels=[];
for l=1:allclass_len-1
    if seen_class_set(2,l)~=0
        num=0;
        for m=1:seen_class_labels_len
            if seen_class_labels(m)==seen_class_set(3,l)
                num=num+1;
                if (num>seen_class_set(2,l))
                    see_train_feature=[see_train_feature;seen_class_feature(m,:)];
                    seen_train_labels=[seen_train_labels;seen_class_labels(m,:)];
                end
                if (num<seen_class_set(2,l))
                    see_test_feature=[see_test_feature;seen_class_feature(m,:)];
                    seen_test_labels=[seen_test_labels;seen_class_labels(m,:)];
                    
                end 
                if (num==seen_class_set(1,l))
                    break;
                end
            end
            
        end
    end
end
save('seen_AWA1.mat','see_train_feature','seen_train_labels','see_test_feature','seen_test_labels')