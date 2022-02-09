
for D in ['amazon']:#,'dslr','webcam']:
    for N in ['noisycorrupted']:
    # for N in ['corrupted','noisy','noisycorrupted']:
        a=open(D+'_list.txt').readlines()
        b=open(D+'_list_WEIGHT.txt').readlines()
        c=open(D+'_list_Relabel_0.8.txt').readlines()
        # c=open(D+'_list_'+N+'_0.4_ALL.txt').readlines()
        A={}
        I=0
        for i in a:
            A[i.split()[0]]=I
            I+=1
        C=open(D+'_list_'+N+'_0.2_Relabel555.txt','w')
        # C=open(D+'_list_'+N+'_0.4_Relabel666.txt','w')
        for i in c:
            C.write(i.strip()+' '+b[A[i.split()[0]]])
        C.close()

