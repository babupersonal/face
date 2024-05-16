function login(){
    num1 = document.getElementById("account_num").value;
    num2 = document.getElementById("password_num").value;
    if(num1 === '1234' && num2 === '1234'){
        window.location.href = '/index'; 
    } else {
        alert('登入失敗');
    }
}
