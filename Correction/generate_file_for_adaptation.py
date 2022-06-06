import os


for D in ['amazon','dslr','webcam']:
    # for N in ['noisycorrupted']:
    for N in ['corrupted','noisy','noisycorrupted']:
        os.system("cp  office_list/"+D+'_list_'+N+'_0.4_left_0.8.txt ../Adaptation/data/office_list/') 
        a=open("office_list/"+D+'_list_'+N+'_0.4.txt').readlines()
        b=open("office_list/"+D+'_list_'+N+'_0.4_WEIGHT.txt').readlines()
        c=open("office_list/"+D+'_list_'+N+'_0.4_Relabel_0.8.txt').readlines()
        A={}
        I=0
        for i in a:
            A[i.split()[0]]=I
            I+=1
        C=open("../Adaptation/data/office_list/"+D+'_list_'+N+'_0.4_Relabel_0.8.txt','w')
        for i in c:
            C.write(i.strip()+' '+b[A[i.split()[0]]])
        C.close()

