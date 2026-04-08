\section*{Tiêu đề nghiên cứu}
\textbf{Phân đoạn ngữ nghĩa nhẹ nhận thức texture cho ảnh món ăn fine-grained}

\section*{Tóm tắt}
Nghiên cứu này đề xuất khảo sát và phát triển một hướng tiếp cận phân đoạn ngữ nghĩa nhẹ nhận thức texture cho ảnh món ăn fine-grained, với thực nghiệm trên tập dữ liệu FoodSeg103. Bài toán phân đoạn thực phẩm ở mức pixel có độ khó cao do các thành phần nguyên liệu thường chồng lấp, ranh giới không rõ ràng, biến thiên lớn trong cùng lớp, trong khi sự khác biệt giữa các lớp lại chủ yếu nằm ở các đặc trưng cục bộ như texture, màu sắc và cấu trúc bề mặt. Trong các hướng tiếp cận hiện nay, các mô hình phân đoạn CNN-based hiệu năng cao thường khai thác tốt ngữ cảnh đa tỉ lệ nhưng có chi phí tính toán lớn, còn các mô hình lightweight cải thiện hiệu quả suy luận nhưng thường làm suy giảm khả năng biểu diễn các đặc trưng chi tiết.

Từ khoảng trống đó, nghiên cứu tập trung làm rõ mối quan hệ giữa hiệu quả tính toán và khả năng bảo toàn đặc trưng texture trong phân đoạn ảnh thực phẩm fine-grained. Trên cơ sở các kiến trúc CNN-based nhẹ, nghiên cứu xây dựng quy trình thực nghiệm nhằm đánh giá ảnh hưởng của backbone, độ phân giải đầu vào và các cơ chế tăng cường đặc trưng cục bộ đến chất lượng phân đoạn. Hiệu năng mô hình sẽ được đánh giá bằng các chỉ số mIoU, mAcc, aAcc, độ trễ suy luận và kích thước mô hình. Kết quả kỳ vọng sẽ cung cấp bằng chứng thực nghiệm về mức độ phù hợp của các hướng tiếp cận lightweight đối với bài toán phân đoạn ảnh thực phẩm, đồng thời xác định một cấu hình khả thi cho các kịch bản triển khai trên thiết bị có tài nguyên hạn chế.

\section*{Bối cảnh nghiên cứu}
Phân đoạn ngữ nghĩa là bài toán gán nhãn cho từng pixel trong ảnh, đòi hỏi mô hình vừa nắm bắt được ngữ cảnh toàn cục vừa bảo toàn được thông tin không gian cục bộ. Trong thị giác máy tính, các hướng tiếp cận hiện đại cho bài toán này chủ yếu dựa trên mạng tích chập sâu (CNN) với các cơ chế downsampling để mở rộng receptive field và các mô-đun tổng hợp đặc trưng đa tỉ lệ nhằm tăng khả năng biểu diễn ngữ nghĩa.

Tuy nhiên, quá trình trừu tượng hóa đặc trưng trong CNN thường làm suy giảm thông tin tần số cao, đặc biệt là biên và texture. Hạn chế này trở nên nghiêm trọng trong các bài toán phân đoạn fine-grained, nơi tín hiệu phân biệt giữa các lớp không nằm chủ yếu ở hình dạng tổng thể mà ở các biến thiên cục bộ của bề mặt đối tượng.

Phân đoạn ảnh thực phẩm là một trường hợp điển hình của bài toán fine-grained. So với các bối cảnh phân đoạn thông thường, ảnh món ăn có các đặc điểm làm gia tăng độ khó: các thành phần nguyên liệu thường chồng lấp, ranh giới giữa các vùng không rõ ràng, hình thái của cùng một lớp biến thiên mạnh theo cách chế biến, và nhiều lớp khác nhau có độ tương đồng thị giác cao. Trong bối cảnh đó, texture giữ vai trò quan trọng trong việc phân biệt các lớp nguyên liệu có đặc điểm hình học không ổn định nhưng khác nhau ở cấu trúc bề mặt hoặc mô thức cục bộ \cite{zhuang2020transfer}.

Tập dữ liệu \textbf{FoodSeg103} được xây dựng như một benchmark cho bài toán phân đoạn ảnh thực phẩm ở mức ingredient với 103 lớp và nhãn pixel-wise. Đây là một tập dữ liệu có số lượng lớp lớn và độ đa dạng cao, phù hợp để đánh giá các mô hình phân đoạn trong điều kiện fine-grained. Đồng thời, tính chất chồng lấp đối tượng và sự tương đồng giữa các lớp khiến FoodSeg103 trở thành một bối cảnh kiểm thử phù hợp cho các nghiên cứu về khả năng bảo toàn đặc trưng cục bộ trong mô hình phân đoạn.

Trong thực tế, nhu cầu tự động hóa trong chuỗi xử lý và giám sát thực phẩm đã thúc đẩy việc phát triển các hệ thống thị giác máy có thể nhận diện thành phần thực phẩm chính xác và hiệu quả \cite{mahalik2010trends, stemmer2018vision, gao2019smartfridge, wang2019smartrefrigerator}. Tuy nhiên, các ứng dụng này thường đi kèm ràng buộc về thời gian suy luận và tài nguyên phần cứng, từ đó đặt ra nhu cầu đối với các mô hình lightweight.

Các hướng tiếp cận CNN-based hiệu năng cao có thể đạt chất lượng phân đoạn tốt nhờ khả năng khai thác ngữ cảnh đa tỉ lệ, nhưng thường đi kèm chi phí tính toán lớn. Ngược lại, các kiến trúc lightweight giảm số lượng tham số và độ phức tạp suy luận để phù hợp hơn với triển khai thực tế, song thường làm suy yếu khả năng biểu diễn các đặc trưng cục bộ do giảm độ sâu mạng hoặc giảm độ phân giải đặc trưng sớm \cite{he2015residual, dong2020mobilenetv2}. Khoảng trống nghiên cứu vì vậy nằm ở việc làm thế nào để duy trì hiệu quả tính toán của mô hình nhẹ mà vẫn bảo toàn và tăng cường được thông tin texture cần thiết cho bài toán phân đoạn ảnh món ăn fine-grained.

\section*{Vấn đề nghiên cứu}
Vấn đề trung tâm của nghiên cứu là sự đánh đổi giữa hiệu quả tính toán và khả năng biểu diễn đặc trưng cục bộ trong các mô hình phân đoạn ngữ nghĩa lightweight. Đối với bài toán phân đoạn ảnh món ăn fine-grained, sự đánh đổi này trở nên rõ nét hơn do các lớp nguyên liệu thường khó phân biệt nếu mô hình không khai thác tốt texture và chi tiết biên.

Mặc dù các hướng tiếp cận CNN-based nhẹ phù hợp với mục tiêu triển khai trên thiết bị có tài nguyên hạn chế, mức độ phù hợp của chúng đối với FoodSeg103 vẫn chưa được làm rõ đầy đủ. Cần xác định liệu các kiến trúc lightweight có thể duy trì chất lượng phân đoạn chấp nhận được trên một benchmark nhiều lớp, giàu texture và có tính fine-grained cao như FoodSeg103 hay không, và trong trường hợp có suy giảm hiệu năng, mức suy giảm đó liên quan như thế nào đến khả năng biểu diễn đặc trưng texture.

\section*{Câu hỏi nghiên cứu}
\textbf{Câu hỏi chính:}  
Các mô hình phân đoạn ngữ nghĩa CNN-based nhẹ có thể đạt được sự cân bằng hợp lý giữa hiệu quả tính toán và chất lượng phân đoạn trên tập dữ liệu FoodSeg103 hay không?

\textbf{Các câu hỏi cụ thể:}
\begin{enumerate}
    \item Các kiến trúc CNN-based nhẹ khác nhau ảnh hưởng như thế nào đến chất lượng phân đoạn trên các lớp thực phẩm có tính fine-grained cao?
    \item Mức độ suy giảm hiệu năng của các mô hình lightweight có liên quan như thế nào đến việc mất thông tin texture và chi tiết cục bộ?
    \item Việc điều chỉnh backbone, độ phân giải đầu vào và cơ chế tăng cường đặc trưng cục bộ có cải thiện được sự đánh đổi giữa độ chính xác và tốc độ suy luận hay không?
    \item Cấu hình mô hình nào phù hợp nhất cho bài toán phân đoạn ảnh thực phẩm trong điều kiện tài nguyên tính toán hạn chế?
\end{enumerate}

\section*{Mục tiêu nghiên cứu}
\textbf{Mục tiêu tổng quát:}  
Khảo sát và đánh giá các hướng tiếp cận phân đoạn ngữ nghĩa CNN-based nhẹ cho ảnh món ăn fine-grained, đồng thời phân tích vai trò của đặc trưng texture trong sự đánh đổi giữa hiệu năng phân đoạn và hiệu quả tính toán.

\textbf{Các mục tiêu cụ thể:}
\begin{enumerate}
    \item Xây dựng quy trình thực nghiệm cho bài toán phân đoạn ngữ nghĩa trên tập dữ liệu FoodSeg103.
    \item Lựa chọn và triển khai một số kiến trúc CNN-based nhẹ làm mô hình đối sánh cho bài toán.
    \item Đánh giá ảnh hưởng của backbone và độ phân giải đầu vào đến các chỉ số chất lượng phân đoạn và chi phí suy luận.
    \item Khảo sát khả năng cải thiện hiệu năng thông qua các cơ chế tăng cường đặc trưng cục bộ theo hướng texture-aware.
    \item Đề xuất một cấu hình mô hình khả thi cho các bối cảnh triển khai trên thiết bị có tài nguyên hạn chế.
\end{enumerate}

\section*{Tổng quan nghiên cứu liên quan}
Các nghiên cứu về phân đoạn ngữ nghĩa hiện đại chủ yếu phát triển theo hai hướng chính. Hướng thứ nhất tập trung vào nâng cao chất lượng biểu diễn ngữ nghĩa thông qua các mô hình CNN-based có khả năng tổng hợp ngữ cảnh đa tỉ lệ. Hướng này thường đạt hiệu năng cao nhưng đi kèm chi phí tính toán lớn. Hướng thứ hai tập trung vào thiết kế các kiến trúc lightweight nhằm giảm số tham số, giảm FLOPs và tăng tốc độ suy luận để phục vụ triển khai thực tế trên thiết bị di động hoặc thiết bị nhúng.

Trong bối cảnh ảnh thực phẩm, các nghiên cứu trước đây cho thấy đây là miền dữ liệu có độ phức tạp cao, nơi sự khác biệt giữa các lớp thường không được xác định rõ bởi hình dạng tổng thể mà bởi các đặc trưng cục bộ như texture, màu sắc và trạng thái bề mặt \cite{zhuang2020transfer}. Điều này khiến các chiến lược giảm độ phân giải đặc trưng quá sớm trong mô hình nhẹ có nguy cơ làm mất thông tin phân biệt quan trọng.

Do đó, một hướng nghiên cứu hợp lý là kết hợp ưu thế về hiệu quả tính toán của kiến trúc lightweight với các cơ chế giúp tăng cường biểu diễn đặc trưng cục bộ. Trong phạm vi nghiên cứu này, các mô hình CNN-based nhẹ được xem như nền tảng thực nghiệm, còn hướng phát triển theo texture-aware được xem như giả thuyết cần kiểm chứng về mặt thực nghiệm trên FoodSeg103.

\section*{Đóng góp nghiên cứu kỳ vọng}
Nghiên cứu kỳ vọng mang lại ba đóng góp chính. Thứ nhất, cung cấp một đánh giá có hệ thống về mức độ phù hợp của các mô hình phân đoạn ngữ nghĩa CNN-based nhẹ đối với bài toán phân đoạn ảnh món ăn fine-grained trên FoodSeg103. Thứ hai, làm rõ hơn mối quan hệ giữa hiệu quả tính toán và khả năng biểu diễn đặc trưng texture trong các mô hình lightweight. Thứ ba, đề xuất một cấu hình hoặc hướng cải tiến theo texture-aware có tính khả thi cho các kịch bản triển khai thực tế trên thiết bị có tài nguyên hạn chế.

\section*{Phương pháp nghiên cứu}
\textbf{Hình thức nghiên cứu:} nghiên cứu thực nghiệm.

\textbf{Dữ liệu:} nghiên cứu sử dụng tập dữ liệu \textbf{FoodSeg103}, gồm 7.118 ảnh được gán nhãn phân đoạn ở mức pixel cho 103 lớp nguyên liệu. Dữ liệu được chia thành 4.983 ảnh huấn luyện và 2.135 ảnh đánh giá. Các bước tiền xử lý dự kiến bao gồm chuẩn hóa kích thước ảnh, cắt ảnh ngẫu nhiên, lật ngang và biến đổi màu để tăng khả năng tổng quát hóa của mô hình.

\textbf{Thiết kế thực nghiệm:} nghiên cứu xây dựng một pipeline thống nhất cho các mô hình CNN-based nhẹ, trong đó thay đổi có kiểm soát các thành phần chính gồm backbone, độ phân giải đầu vào và mô-đun tăng cường đặc trưng cục bộ. Cách thiết kế này nhằm tách biệt ảnh hưởng của từng yếu tố đến chất lượng phân đoạn và hiệu quả suy luận.

\textbf{Hướng tiếp cận mô hình:} thay vì tập trung vào một kiến trúc cụ thể, nghiên cứu khảo sát nhóm các hướng tiếp cận CNN-based cho phân đoạn ngữ nghĩa, bao gồm cả cấu hình cơ sở và các biến thể lightweight. Trên nền các mô hình này, nghiên cứu sẽ thử nghiệm các cơ chế tăng cường đặc trưng cục bộ theo hướng texture-aware, ví dụ thông qua tinh chỉnh nhánh chi tiết, hợp nhất đặc trưng hoặc cải thiện biểu diễn biên.

\textbf{Đánh giá:} hiệu năng mô hình được đánh giá bằng các chỉ số \textit{mean Intersection over Union} (mIoU), \textit{mean Accuracy} (mAcc), \textit{all Accuracy} (aAcc), độ trễ suy luận, kích thước mô hình và chi phí tính toán. Kết quả sẽ được phân tích theo hướng đánh đổi giữa chất lượng phân đoạn và khả năng triển khai.

\textbf{Giới hạn và tính khả thi:} nghiên cứu giới hạn trong các mô hình CNN-based nhẹ và một benchmark chính là FoodSeg103. Việc lựa chọn phạm vi này giúp đảm bảo tính khả thi về thời gian, tài nguyên tính toán và khả năng so sánh công bằng giữa các mô hình.

\section*{Kế hoạch thực hiện}
\begin{center}
\begin{tabular}{|c|p{10cm}|}
\hline
\textbf{Thời gian} & \textbf{Nội dung} \\
\hline
Tuần 1 & Khảo sát tài liệu, xác lập khoảng trống nghiên cứu, chuẩn bị dữ liệu và môi trường thực nghiệm. \\
\hline
Tuần 2 & Xây dựng pipeline huấn luyện và đánh giá cơ sở trên FoodSeg103. \\
\hline
Tuần 3 & Triển khai và so sánh các kiến trúc CNN-based nhẹ với các cấu hình backbone và độ phân giải đầu vào khác nhau. \\
\hline
Tuần 4 & Thử nghiệm các cơ chế tăng cường đặc trưng cục bộ theo hướng texture-aware, phân tích kết quả và rút ra cấu hình phù hợp. \\
\hline
Tuần 5 & Tổng hợp kết quả, hoàn thiện báo cáo và thảo luận đóng góp nghiên cứu. \\
\hline
\end{tabular}
\end{center}

\section*{Tài nguyên cần thiết}
\textbf{Dữ liệu:} FoodSeg103.  

\textbf{Phần cứng:} GPU có bộ nhớ từ 16GB VRAM trở lên.  

\textbf{Phần mềm:} PyTorch và các công cụ phục vụ đánh giá, tối ưu suy luận và triển khai mô hình.

\section*{Tài liệu tham khảo}
\bibliographystyle{plain}
\bibliography{references}