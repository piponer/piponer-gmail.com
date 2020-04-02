
1, need to install package and create virtual environment follow requirements.txt
 
2, sample input tranning data download the files from https://drive.google.com/open?id=1iqKwXqKZC1kHCJLyEgT3mpSY5agnQkTX and drag them into the app directory, glove.6B, useful_dataset is not necessary to run the application but it needed for training.

3, install cuda 10.2 and cudnn 7.6 (optional)
   Can cuda support your display card, you can check on the https://bbs.csdn.net/topics/390793022 

4,  install pytorch from https://pytorch.org/get-started/locally/ (Note: we used pytorch==1.1.0 and torchvision==0.4.1)
    and you can check the pytorch on the https://pytorch.org/
    https://www.youtube.com/watch?v=QJgjcnuQqNI&list=PLgAyVnrNJ96CqYdjZ8v9YjQvCBcK5PZ-V  
    https://www.pytorchtutorial.com/5-2-gpu-in-pytorch/
    
5,  install the MySQL and create the database, set USERNAME and PASSWORD
    
6, open app/init.py and set USERNAME and PASSWORD to your MySQL username and password

7, To run the program: python run.py
