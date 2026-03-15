\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uitproposal}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions\relax
\LoadClass[14pt,a4paper]{memoir}

% Require package
\RequirePackage{anyfontsize,mathptmx}
\RequirePackage[T5]{fontenc}
\RequirePackage{geometry,graphicx,indentfirst,setspace}
\RequirePackage{tikz}
\RequirePackage[center]{caption}
\RequirePackage{fancyhdr,hyperref}
\hypersetup{urlcolor=blue,linkcolor=black,citecolor=black,colorlinks=true} 
\RequirePackage[vietnamese,english]{babel}
\usetikzlibrary{calc}

% Renew, new command
\renewcommand{\@title}{}
\newcommand{\@vntitle}{}
\newcommand{\vntitle}[1]{\renewcommand{\@vntitle}{#1}}
\newcommand{\@proposaltype}{}
\newcommand{\proposaltype}[1]{\renewcommand{\@proposaltype}{#1}}
\newcommand{\@supervisorname}{}
\newcommand{\supervisorname}[1]{\renewcommand{\@supervisorname}{#1}}
\newcommand{\@studentname}{}
\newcommand{\studentname}[1]{\renewcommand{\@studentname}{#1}}

\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\arabic{subsection}}

% \patchcmd{\thebibliography}{\chapter*}{\section*}{}{}
\def\headerbox {
    \begin{tikzpicture}[remember picture,overlay]
        \draw[line width=3pt](
        $(current page.north west) + (0.7cm,-0.7cm)$
        ) rectangle (
        $(current page.south east) + (-0.7cm,0.7cm)$
        );
        \draw[line width=1pt](
        $(current page.north west) + (1cm,-1cm)$
        ) rectangle (
        $(current page.south east) + (-1cm,1cm)$
        );
      \end{tikzpicture}
}

\geometry{top=1cm, bottom=1cm, left=0.7cm, right=1cm}
\def\enproposalheader{
    \pagestyle{empty}
    \begin{center}
        \begin{tabular}{ccc}
            % \includegraphics[width=0.1\textwidth]{imgs/vnu_logo.png} &
            \begin{minipage}[b]{0.8\textwidth}
                \centering
                \textbf{VIETNAM NATIONAL UNIVERSITY, HO CHI MINH CITY} \\
                \textbf{UNIVERSITY OF SCIENCE} 
            \end{minipage} &
            % \includegraphics[width=0.08\textwidth]{imgs/UIT-logo.png} \\
        \end{tabular}
    \end{center}
    \vspace{2cm}
}

\def\vnproposalheader{
    \pagestyle{empty}
    \begin{center}
        \begin{tabular}{ccc}
            % \includegraphics[width=0.12\textwidth]{imgs/vnu_logo.png} &
            \begin{minipage}[b]{0.7\textwidth}
                \centering
                \textbf{ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH} \\
                \textbf{TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN} 
            \end{minipage} &
            % \includegraphics[width=0.1\textwidth]{imgs/UIT-logo.png} \\
        \end{tabular}
    \end{center}
    \vspace{2cm}
    
}

\geometry{paper=a4paper,top=2cm, bottom=1.5cm, left=2cm, right=2cm}

\def\vnproposalname {
    \begin{center}
        \Large\textbf{ĐỀ CƯƠNG NGHIÊN CỨU} \\
        \large\MakeUppercase{\textbf{\@proposaltype}} \\
        \vspace{3cm}
        {\large{\centering{\textbf{Tên đề tài: \@vntitle}}}} \\
        \vspace{1cm}
        % {\large{\centering\MakeUppercase{\textbf{Tên đề tài tiếng Anh: \@title}}}}
        \vfill
        {\large{\centering{\textbf{Giảng viên hướng dẫn} \\ \@supervisorname}}} \\
        \vspace{3cm}
        {\large{\centering{\textbf{Học viên thực hiện} \\ \@studentname}}}
        \vfill
        \today
    \end{center}
}

\def\enproposalname {
    \begin{center}
        \Large\textbf{RESEARCH PROPOSAL} \\
        \large\MakeUppercase{\textbf{\@proposaltype}} \\
        \vspace{3cm}
        % {\large{\centering\MakeUppercase{\textbf{Tên đề tài: \@vntitle}}}} \\
        % \vspace{1cm}
        {\large{\centering\MakeUppercase{\textbf{RESEARCH TOPIC: \@title}}}}
        \vfill
        {\large{\centering\MakeUppercase{\textbf{Instructor \\ \@supervisorname}}}} \\
        \vspace{3cm}
        {\large{\centering\MakeUppercase{\textbf{Student \\ \@studentname}}}}
        \vfill
        \today
    \end{center}
}
