clear
%%%%%%%%%%%%%%%%%%%%%获取训练类与测试类的2048维视觉特征与312维语义特征%%%%%%%%%%%%%%%%%%%%%
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
%%只有AWA1是-1，其他的数据不用-1（最后一个类别不是测试类）
for k=1:allclass_len-1
    seen_class_set(2,k)=seen_class_set(1,k)/11;
    seen_class_set(3,k)=k;
end
seen_class_feature=trainclasses_features;
%%%%遍历3*49的set数组，各类别第二行取出来转换为int不为0的话，就
%%%%遍历features及labels，如果属于该类的话，取对应set中第二行个，存入新的seen_test中，剩下的存入seen_train中
%%%%新建seen_test_labels和seen_train_labels，与上述对应
%%%%将建好的4个seen_test、seen_train、seen_test_labels、seen_train_labels文件存入新的mat文件
%%%%Python分别读入，训练与未见类生成的拼接，即可训练和预测
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