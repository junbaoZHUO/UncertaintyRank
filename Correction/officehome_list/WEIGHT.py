
for D in ['Art','Clipart','Product', 'Real_world']:
    # for N in ['corrupted','noisy','noisycorrupted']:
    for N in ['noisycorrupted']:
        a=open(D+'_list_'+N+'_0.4.txt').readlines()
        b=open(D+'_list_'+N+'_0.4_WEIGHT.txt').readlines()
        c=open(D+'_list_'+N+'_0.4_Relabel.txt').readlines()
        A={}
        I=0
        for i in a:
            A[i.split()[0]]=I
            I+=1
        C=open(D+'_list_'+N+'_0.4_Relabel2.txt','w')
        for i in c:
            C.write(i.strip()+' '+b[A[i.split()[0]]])
        C.close()

