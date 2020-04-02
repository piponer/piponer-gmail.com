# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:08:22 2019

@author: 63184
"""

import json
import os
from matplotlib.pylab import *

def json_default_type_checker(o):
    """
    From https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
    """
    if isinstance(o, int64): return int(o)
    raise TypeError

def save_for_evaluation(path_save, results, dset_name):
    path_save_file = os.path.join(path_save, f'results_{dset_name}.json')
    with open(path_save_file, 'w', encoding='utf-8') as f:
        for i, r1 in enumerate(results):
            json_str = json.dumps(r1, ensure_ascii=False, default=json_default_type_checker)
            json_str += '\n'

            f.writelines(json_str)
            
def single_course(code, title, url, lecturer, admin, summary):
    cur_dict = dict()
    cur_dict['course_code'] = code
    cur_dict['course_title'] = title
    cur_dict['course_url'] = url
    cur_dict['lecturer'] = lecturer
    cur_dict['admin'] = admin
    cur_dict['summary'] = summary
    cur_dict['credit'] = 6
    cur_dict['faculty'] = 'Faculty of Engineering'
    cur_dict['school'] = 'School of Computer Science and Engineering'
    cur_dict['campus'] = 'Kensington'
    cur_dict['hours'] = 48
    cur_dict['course_fee'] = 5970
    return cur_dict

if __name__ == '__main__':
    results = []
    results.append(single_course("COMP1521", "Computer Systems Fundamentals", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP1521/?q=comp1521&ct=all", "Dr Andrew Taylor", "Jashank Jeremy", "This course provides a programmer's view on how a computer system executes programs, manipulates data and communicates. It enables students to become effective programmers in dealing with issues of performance, portability, and robustness. It is typically taken in the semester after completing COMP1511, but could be delayed and taken later. It serves as a foundation for later courses on networks, operating systems, computer architecture and compilers, where a deeper understanding of systems-level issues is required."))
    results.append(single_course("COMP3331", "Computer Networks and Applications", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP3331/?q=COMP3331&ct=all", "Salil Kanhere", "Nadeem Ahmed", "Networking technology overview. Protocol design and validation using the finite state automata in conjunction with time-lines. Overview of the IEEE802 network data link protocol standards. Addressing at the data link and network layers. Network layer services. Introduction to routing algorithms such as Distance Vector and Link State. Congestion control mechanisms. Internetworking issues in connecting networks. The Internet Protocol Suite overview. The Internet protocols IPv4 and IPv6. Address resolution using ARP and RARP. Transport layer: issues, transport protocols TCP and UDP. Application level protocols such as: File Transfer Protocol (FTP), Domain Name System (DNS) and Simple Mail Transfer Protocol (SMTP). Introduction to fundamental network security concepts, 802.11 wireless networks and peer to peer networks. There is a substantial network programming component in the assessable material."))
    results.append(single_course("COMP2521", "Data Structures and Algorithms", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp2521/?q=comp2521", "Ashesh Mahidadia", "Mei Cheng Whale", "The goal of this course is to deepen students' understanding of data structures and algorithms and how these can be employed effectively in the design of software systems. We anticipate that it will generally be taken in the second year of a program, but since its only pre-requisite is COMP1511, is it possible to take it in first year. It is an important course in covering a range of core data structures and algorithms that will be used in context in later courses."))
    results.append(single_course("COMP1511", "Introduction to Programming", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp1511/?q=comp1511", "Marc Chee", "Andrew Bennett", "An introduction to problem-solving via programming, which aims to have students develop proficiency in using a high level programming language. Topics: algorithms, program structures (statements, sequence, selection, iteration, functions), data types (numeric, character), data structures (arrays, tuples, pointers, lists), storage structures (memory, addresses), introduction to analysis of algorithms, testing, code quality, teamwork, and reflective practice. The course includes extensive practical work in labs and programming projects."))
    results.append(single_course("COMP9321", "Data Services Engineering", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9321/?q=comp9321", "Morty Al-Banna", "Mohammadali Yaghoubzadehfard", "This course aims to introduce the student to core concepts and practical skills for engineering the data in Web-service-oriented data-driven applications. Specifically, the course aims to expose students to basic infrastructure for building data services on the Web, including techniques to access and ingest data in internal/external sources, develop software services to curate (e.g. extract, transform, correct, aggregate the data), develop services to apply various analytics and develop services to visualize the data to communicate effectively using data. The course uses the Python programming language as the practical basis for its modules. However, the concepts taught are universal and can be applied to any other web development language/framework."))
    results.append(single_course("COMP9311", "Database Systems", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9311/?q=comp9311", "Wenjie Zhang", "Kai Wang", "A first course on database management systems. Data modelling; principles of database design; data manipulation languages; database application techniques; introduction to DBMS internals; introduction to advanced databases. Lab: design and implementation of a database application using SQL and stored procedures."))
    results.append(single_course("COMPMP2511", "Object-Oriented Design & Programming", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp2511/?q=comp2511", "Ashesh Mahidadia", "Robert Clifton-Everest", "This course aims to introduce students to the principles of object-oriented design and to fundamental techniques in object-oriented programming. It is typically taken in the second year of study, after COMP2521, to ensure an appropriate background in data structures. The knowledge gained in COMP2511 is useful in a wide range of later-year CS courses."))
    results.append(single_course("COMP9020", "Foundations of Computer Science", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9020/?q=comp9020", "Paul Hunter", "Paul Hunter", "Scope: * Mathematical methods for designing correct and efficient programs.* Mathematics for algorithm analysis.* Logic for proving and verification.Topics: * Introduction to set and relation theory* Propositional logic and boolean algebras* Induction, recursion and recurrence relations* Order of growth of functions.* Structured counting (combinatorics)* Discrete probability* Graph theory* Trees for algorithmic applications"))
    results.append(single_course("COM9444", "Neural Networks and Deep Learning", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9444/?q=comp9444", "Alan Blair", "Alexander Long", "Topics chosen from: perceptrons, feedforward neural networks, backpropagation, Hopfield and Kohonen networks, restricted Boltzmann machine and autoencoders, deep convolutional networks for image processing; geometric and complexity analysis of trained neural networks; recurrent networks, language processing, semantic analysis, long short term memory; designing successful applications of neural networks; recent developments in neural networks and deep learning."))
    results.append(single_course("COMP3311", "Database Systems", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP3311/?q=comp3311&ct=all", "John Shepherd", "Hayden Smith", "Data models: entity-relationship, relational, object-oriented. Relational database management systems: data definition, query languages, development tools. Database application design and implementation. Architecture of relational database management systems: storage management, query processing, transaction processing. Lab: design and implementation of a database application."))
    results.append(single_course("COMP1531", "Software Engineering Fundamentals", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp1531/?q=comp1531", "Hayden Smith", "Robert Clifton-Everest", "This course provides an introduction to software engineering principles: basic software lifecycle concepts, modern development methodologies, conceptual modeling and how these activities relate to programming. It also introduces the basic notions of team-based project management via conducting a project to design, build and deploy a simple web-based application. It is typically taken in the semester after completing COMP1511, but could be delayed and taken later. It provides essential background for the teamwork and project management required in many later courses."))
    results.append(single_course("COP9024", "Data Structures and Algorithms", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/COMP9024/?q=comp9024&ct=all", "Michael Thielscher", "Michael Schofield", "Data types and data structures: abstractions and representations; lists, stacks, queues, heaps, graphs; dictionaries and hash tables; search trees; searching and sorting algorithms."))
    results.append(single_course("COMP9315", "DBMS Implementation", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9315/?q=comp9315", "John Shepherd", "Hayden Smith", "Detailed examination of techniques used in the implementation of relational, object-oriented and distributed database systems. Topics are drawn from: query optimisation, transaction management, advanced file access methods, database performance tuning."))
    results.append(single_course("COMP9313", "Big Data Management", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP9313/?q=comp9313&ct=all", "Morty Al-Banna", "Maisie Badami", "This course introduces the core concepts and technologies involved in managing Big Data. Topics include: characteristics of Big Bata and Big Data analysis, storage systems (e.g. HDFS, S3), techniques for manipulating Big Data (e.g. MapReduce, streaming, compression), programming languages (e.g. Spark, PigLatin), query languages (e.g. Jaql, Hive), database systems (e.g. noSQL systems, HBase), and typical applications (e.g. recommender systems, dimensionality reduction, text analysis)."))
    save_for_evaluation('.', results, 'course1')