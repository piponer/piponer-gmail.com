# mysql>\. d:\comp3900\insert.sql
# mysql>desc course;
# mysql> delete from course where id>=1; 

CREATE TABLE IF NOT EXISTS course (
            id INT NOT NULL PRIMARY KEY AUTO_INCREMENT UNIQUE,  
            course_code TEXT,
            course_title TEXT,
            course_url TEXT,
            lecturer TEXT,
            admin TEXT,
            credit double,
            faculty TEXT,
            school TEXT,
            campus TEXT,
            hours double                  
          )ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP6733',
				'Internet of Things Design Studio',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/COMP6733',
				'Salil Kanhere',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);


INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9024',
				'Data Structures and Algorithms',
				'https://www.handbook.unsw.edu.au/postgraduate/courses/2019/COMP9024/',
				'Michael Schofield, Michael Thielscher ,Hui Wu,Ashesh Mahidadia',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'SENG2011',
				'Software Engineering Workshop 2A',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/SENG2011',
				'Ian Chee-Wing Wong ,Manas Patra ,Ron Van der Meyden',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);
INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP2121',
				'Microprocessors and Interfacing',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/COMP2121/',
				'Hasindu Gamaarachchi ,Hui Wu ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 6
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9900',
				'Information Technology Project',
				'https://www.handbook.unsw.edu.au/postgraduate/courses/2019/COMP9900',
				'Rachid Hamadi,Wael Alghamdi ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 10
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP6741',
				'Parameterized and Exact Computation',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/COMP6741',
				'Shenwei Huang ,Edward Jay Lee ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP1000',
				'Web, Spreadsheets and Databases',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/comp1000/',
				'Wei Xu,Aarthi Natarajan ,Isaac Carr ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP3211',
				'Computer Architecture',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/COMP3211/',
				'Hui Guo ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 6
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP3222',
				'Digital Circuits and Systems',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/COMP3222/',
				'Oliver Diessel ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 7
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP6445',
				'Digital Forensics and Incident Response',
				'https://www.handbook.unsw.edu.au/postgraduate/courses/2019/comp6445/',
				'Tim Boyce ,Roland Wen ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9322',
				'Software Service Design and Engineering',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/COMP9322',
				'Fethi Rabhi ,Onur Demirors',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 3
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP4601',
				'Design Project B',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/COMP4601/',
				'Oliver Diessel ,Alexander Kroh ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 6
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP1400',
				'Programming for Designers',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/comp1400/',
				'Hailun Tan ,Isaac Carr ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 6
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP3431',
				'Robot Software Architectures',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/COMP3431',
				'Timothy Wiley ,David Rajaratnam ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9243',
				'Distributed Systems',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2019/COMP9243',
				'Ihor Kuz ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP1911',
				'Computing 1A',
				'http://cse.unsw.edu.au/~cs1911',
				'Hayden Smith',
				'Hayden Smith',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 7
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP1911',
				'Computing 1A',
				'http://cse.unsw.edu.au/~cs1911',
				'Hayden Smith',
				'Hayden Smith',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 7
				);
INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9020',
				'Foundations of Computer Science',
				'http://www.cse.unsw.edu.au/~cs9020',
				'Paul Hunter',
				'Paul Hunter',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'ENGG1811',
				'Computing for Engineers',
				'http://www.cse.unsw.edu.au/~en1811',
				'null',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9319',
				'Web Data Compression and Search',
				'http://cse.unsw.edu.au/~cs9319',
				'Raymond Wong',
				'Raymond Wong',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9313',
				'Big Data Management',
				'http://cse.unsw.edu.au/~cs9313',
				'null',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 3
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9334',
				'Capacity Planning of Computer Systems and Networks',
				'http://cse.unsw.edu.au/~cs9334',
				'null',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP4418',
				'Knowledge Representation and Reasoning',
				'http://cse.unsw.edu.au/~cs4418',
				'Maurice Pagnucco',
				'Maurice Pagnucco',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9024',
				'Data Structures and Algorithms',
				'http://www.cse.unsw.edu.au/~cs9024',
				'Michael Thielscher',
				'Michael Thielscher',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP1531',
				'Software Engineering Fundamentals',
				'http://cse.unsw.edu.au/~cs1531',
				'Hayden Smith , Rob Everest',
				'Hayden Smith , Rob Everest',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 7
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9517',
				'Computer Vision',
				'http://cse.unsw.edu.au/~cs9517',
				'Prof Arcot Sowmya, Dr Yang Song, Prof Erik Meijering',
				'Annette Spooner',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9417',
				'Machine Learning and Data Mining',
				'http://cse.unsw.edu.au/~cs9417',
				'Michael Bain',
				'Omar Ghattas',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9418',
				'Advanced Topics In Statistical Machine Learning',
				'http://cse.unsw.edu.au/~cs9418',
				'Gustavo Batista',
				'Armin Chitizadeh',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP2521',
				'Data Structures and Algorithms',
				'https://webcms3.cse.unsw.edu.au/COMP2521/19T3/',
				'Ashesh Mahidadia ',
				'null',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 7
				);
INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9032',
				'Microprocessors and Interfacing',
				'http://cse.unsw.edu.au/~cs9032/',
				'Hui Guo',
				'Hui Guo',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP1521',
				'Computer Systems Fundamentals',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP1521/?q=comp1521&ct=all',
				'Dr Andrew Taylor',
				'Jashank Jeremy',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP3331',
				'Computer Networks and Applications',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP3331/?q=COMP3331&ct=all',
				'Salil Kanhere',
				'Nadeem Ahmed',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP2521',
				'Data Structures and Algorithms',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp2521/?q=comp2521',
				'Ashesh Mahidadia',
				'Mei Cheng Whale',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP1511',
				'Introduction to Programming',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp1511/?q=comp1511',
				'Marc Chee',
				'Andrew Bennett',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9321',
				'Data Services Engineering',
				'https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9321/?q=comp9321',
				'Morty Al-Banna',
				'Mohammadali Yaghoubzadehfard',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9311',
				'Database Systems',
				'https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9311/?q=comp9311',
				'Wenjie Zhang',
				'Kai Wang',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMPMP2511',
				'Object-Oriented Design & Programming',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp2511/?q=comp2511',
				'Ashesh Mahidadia',
				'Robert Clifton-Everest',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9020',
				'Foundations of Computer Science',
				'https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9020/?q=comp9020',
				'Paul Hunter',
				'Paul Hunter',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COM9444',
				'Neural Networks and Deep Learning',
				'https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9444/?q=comp9444',
				'Alan Blair',
				'Alexander Long',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP3311',
				'Database Systems',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP3311/?q=comp3311&ct=all',
				'John Shepherd',
				'Hayden Smith',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);
INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP3311',
				'Database Systems',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP3311/?q=comp3311&ct=all',
				'John Shepherd',
				'Hayden Smith',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP1531',
				'Software Engineering Fundamentals',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp1531/?q=comp1531',
				'Hayden Smith',
				'Robert Clifton-Everest',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COP9024',
				'Data Structures and Algorithms',
				'https://www.handbook.unsw.edu.au/postgraduate/courses/2020/COMP9024/?q=comp9024&ct=all',
				'Michael Thielscher',
				'Michael Schofield',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9315',
				'DBMS Implementation',
				'https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9315/?q=comp9315',
				'John Shepherd',
				'Hayden Smith',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 4
				);

INSERT ignore INTO course (
		course_code,course_title,course_url,
		lecturer,admin,credit,faculty,school,campus,hours) 
			VALUES(
				'COMP9313',
				'Big Data Management',
				'https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP9313/?q=comp9313&ct=all',
				'Morty Al-Banna',
				'Maisie Badami',
				 6,
				 'Engineering',
				 'Computer Science and Engineering',
				 'Kensington',
				 5
				);