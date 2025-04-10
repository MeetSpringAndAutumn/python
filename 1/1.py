using System.Data.SqlClient;
protected void Page_Load(object sender , EventArgs e)
{
    if (Session["AID"] != null || Session["UserID"] != null)
    {
        if (!Page.IsPostBack)
        {
            SqlConnection conn= new SqlConnection();
            conn.ConnectionString = "Data Source=.\\SQLEXPRESS;AttachDbFilename=C:\\Inetpub\\wwwroot\\WebMall\\App_Data\\mall.mdf;Integrated Security=True;User Instance=True";
            conn.Open();
            SqlCommand cmd = conn.CreateCommand();
            // 记录不同类别的用户
            if (Session["UserID"] != null)
            {
                cmd.CommandText = "SELECT UserID, UserName FROM UserTable where UserID=" + Session["UserID"].ToString();
            }
            else
            {
                cmd.CommandText = "SELECT UserID,UserName FROM UserTable where UserID=" + Session["AID"].ToString();
            }
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            Dataset ds = new Dataset();
            da.Fill(ds, "UserTable");
            conn.Close();
            Label1.Text = ds.Tables[0].Rows[0]["UserName"].ToString().Trim();
        }
    }
}
