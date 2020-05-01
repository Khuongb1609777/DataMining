<?php 
	include('connect.php');
	$con -> set_charset('utf8');
	$_listst=array(
		1 => "thể thao",
		2 => "du lịch",
		3 => "âm nhạc",
		4 => "thời trang"
	);

    if(isset($_POST['submit'])){
    	if(empty($_POST['txttendangnhap'])
    		or empty($_POST['txtmatkhau'])
    		or empty($_POST['txtgolaimatkhau'])
    		or empty($_FILES['slcfile'])
    		or empty($_POST['rdgioitinh'])
    		or empty($_POST['slsnghenghiep'])
    		or empty($_POST['sothich']))	
    	{
    		echo'<p style="color:red">Vui long khong de trong bat ki muc nao</p>';
    	}

    	else if(($_POST['txtmatkhau'])!=($_POST['txtgolaimatkhau']))
    	{
    		echo'<p style="color:red">Mật khẩu không giống nhau</p>';
    	}
    	else 
    	{	
			$list=array();
    		$username = $_POST['txttendangnhap'];//lay du lieu ten dang nhap gan vao bien username
    		$password = $_POST['txtmatkhau'];//lay du lieu mat khau gan vao bien password
    		$password = md5($password);//ma hoa password
    		$gender = $_POST['rdgioitinh'];//lay du lieu gioi tinh gan vao bien gender
    		$job = $_POST['slsnghenghiep'];//lay du lieu nghe nghiep gan vao bien job
    		//lay du lieu so thich bang foreach, ben tren em co khoi tao 1 array $_listst chua cac so thich
    		//ben html em dung value la 1,2,3,4 tuong ung voi cac so thich trong array em khai bao ben tren
    		foreach($_POST['sothich'] as $value){
				//echo $_listst[$value];
				$list[] = $_listst[$value];
				//$sothich = implode ($str, $interest);
				//echo $value."<br>";
			}
			$interest = implode(',',$list);
			echo $interest;
    		//echo "ten dang nhap: $username<br>"; 
    		//echo "$password <br>";
    		//echo "$gender <br>";
    		//echo "$job <br>";	
    		
    		//in ra cac thong tin ve file

    		/*echo"<pre>";
			$img = $_FILES['slcfile'];
			print_r ($img);
			echo"</pre>";
			*/

			// dat co hieu bang mang
			$error = array();
			//tao folder uploads chua file
			$target_dir = "./../Buoi3/uploads/";
			//tao duong dan file uploaded
			$target_file = $target_dir.basename($_FILES['slcfile']['name']);

			//kiem tra dieu kien upload
			$file_name = $_FILES['slcfile']['name'];
			$file_type = $_FILES['slcfile']['type'];
			$file_size = $_FILES['slcfile']['size'];
			$file_type2 = pathinfo($_FILES['slcfile']['name'], PATHINFO_EXTENSION);
			//move_uploaded_file($_FILES['slcfile']['tmp_name'], $target_file);

			/*
			//1-dieu kien ve kich thuoc file vd cho phep 5mb
			if($file_size > 5242880){
					$error['slcfile'] = "Chi duoc up load file duoi 5mb";
			}

			//2-kiem tra loai file
			$file_type_test = array('png', 'jpg', 'jpeg','gif');
			if(!in_array(strtolower($file_type2), $file_type_test)){
				$error['slcfile'] = "chi cho phep file anh (jpg, jpeg, png, gif)";
				}
			
			//3-kiem tra file da co chua
			if(file_exists($target_file)){
				$error['slcfile'] = "file da ton tai";
			}*/
			//chuyen file tu bo nho tam len server

			//move_uploaded_file($_FILES['slcfile']['tmp_name'], $target_file);

			if(empty($error)){
				if (move_uploaded_file($_FILES['slcfile']['tmp_name'], $target_file)){
					//echo $target_file;
					$flag = true;

				}else {echo"upload that bai";
						print_r($error);}

			}
		
			

			$sql = "INSERT INTO thanhvien(tendangnhap,matkhau,hinhanh,gioitinh,nghenghiep,sothich) VALUES ('$username','$password','$target_file','$gender','$job','$interest')";
            $them = mysqli_query($con,$sql);
            
            if($them){
				echo '<p style="color:red">them thanh cong</p>';
				$flag = 1;
                header('Location:dangnhap.html?flag='.$flag);
            }else{echo '<p style="color:red">them khong thanh cong</p>';
            }

			//-------------------------------------------------------------
    	}

    }

    $con -> close();
?>