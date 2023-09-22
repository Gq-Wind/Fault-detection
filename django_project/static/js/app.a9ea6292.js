(function(){"use strict";var e={7406:function(e,t,a){var s=a(9197),r=a(8473);const n={class:""};function l(e,t,a,s,l,o){const c=(0,r.up)("router-view");return(0,r.wg)(),(0,r.iD)("div",n,[(0,r.Wm)(c)])}var o={name:"App",components:{}},c=a(5312);const i=(0,c.Z)(o,[["render",l]]);var d=i,u=a(4731),p=a(4887),m=a.p+"static/img/logo.18e064fb.png";const f=e=>((0,r.dD)("data-v-b0a48e14"),e=e(),(0,r.Cn)(),e),g={class:"loginpage"},v={class:"login"},w=f((()=>(0,r._)("img",{src:m,alt:""},null,-1))),h={class:"topnav"},y={class:"form-group"},b={class:"form-group"},_={key:0,class:"tip2"},k=f((()=>(0,r._)("button",{class:"login-btn",type:"submit"},"登录",-1)));function W(e,t,a,n,l,o){const c=(0,r.up)("router-link");return(0,r.wg)(),(0,r.iD)("div",g,[(0,r._)("div",v,[(0,r.Wm)(c,{to:"/",class:"logo"},{default:(0,r.w5)((()=>[w])),_:1}),(0,r._)("div",h,[(0,r.Wm)(c,{to:"/about",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)("文档")])),_:1}),(0,r.Wm)(c,{to:"/register",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)("注册")])),_:1}),(0,r.Wm)(c,{to:"/",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)("首页")])),_:1})]),(0,r._)("form",{onSubmit:t[2]||(t[2]=(0,s.iM)((function(){return e.login&&e.login(...arguments)}),["prevent"]))},[(0,r._)("div",y,[(0,r.wy)((0,r._)("input",{type:"text","onUpdate:modelValue":t[0]||(t[0]=t=>e.username=t),placeholder:"用户名"},null,512),[[s.nr,e.username]])]),(0,r._)("div",b,[(0,r.wy)((0,r._)("input",{type:"password","onUpdate:modelValue":t[1]||(t[1]=t=>e.password=t),placeholder:"密码"},null,512),[[s.nr,e.password]]),e.errorMessage?((0,r.wg)(),(0,r.iD)("div",_,(0,p.zw)(e.errorMessage),1)):(0,r.kq)("",!0)]),k,(0,r._)("p",null,[(0,r.Uk)("没有账号？去"),(0,r.Wm)(c,{to:"/register",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)(" 注册")])),_:1})])],32)])])}a(7658);var U=a(4188),C=a(9868),N=a.n(C),A=(0,r.aZ)({setup(){const e=(0,U.iH)(""),t=(0,U.iH)(""),a=(0,U.iH)(""),s=new FormData,r=async()=>{try{s.append("username",e.value),s.append("password",t.value);const a=await N().post("user/login/",s),r=a.headers.authorization.split(" ")[1];localStorage.removeItem("username"),localStorage.removeItem("token"),localStorage.setItem("username",e.value),localStorage.setItem("token",r),Qt.push({path:"/workspace"})}catch(r){const e=r.response.data.error_msg;a.value=e}};return{username:e,password:t,errorMessage:a,login:r}}});const P=(0,c.Z)(A,[["render",W],["__scopeId","data-v-b0a48e14"]]);var D=P;const x=e=>((0,r.dD)("data-v-3c0d47d2"),e=e(),(0,r.Cn)(),e),M={class:"homepage"},S={class:"home"},V=x((()=>(0,r._)("img",{src:m,alt:""},null,-1))),T={class:"topnav"},j=x((()=>(0,r._)("h1",null,"Startup",-1))),q=x((()=>(0,r._)("div",{class:"introduction"},[(0,r.Uk)(" 分布式故障分类系统，"),(0,r._)("br"),(0,r.Uk)(" 接入多种神经网络和机器学习分布式故障分类算法，"),(0,r._)("br"),(0,r.Uk)(" 也可训练自己的专属模型，更快更精准！ ")],-1)));function F(e,t){const a=(0,r.up)("router-link");return(0,r.wg)(),(0,r.iD)("div",M,[(0,r._)("div",S,[(0,r.Wm)(a,{to:"/",class:"logo"},{default:(0,r.w5)((()=>[V])),_:1}),(0,r._)("div",T,[(0,r.Wm)(a,{to:"/about",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)("文档")])),_:1}),(0,r.Wm)(a,{to:"/register",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)("注册")])),_:1}),(0,r.Wm)(a,{to:"/login",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)("登录")])),_:1})]),j,q,(0,r.Wm)(a,{to:"/workspace",class:"login"},{default:(0,r.w5)((()=>[(0,r.Uk)("创建项目")])),_:1})])])}const z={},I=(0,c.Z)(z,[["render",F],["__scopeId","data-v-3c0d47d2"]]);var H=I;const O=e=>((0,r.dD)("data-v-bbb8d43e"),e=e(),(0,r.Cn)(),e),X={class:"registerpage"},B={class:"register"},J=O((()=>(0,r._)("img",{src:m,alt:""},null,-1))),R={class:"topnav"},Y={class:"form-group"},L={class:"form-group"},Q={class:"form-group"},E={class:"form-group"},Z=["src"],K=O((()=>(0,r._)("button",{class:"register-btn",type:"submit"},"注册",-1)));function G(e,t,a,n,l,o){const c=(0,r.up)("router-link");return(0,r.wg)(),(0,r.iD)("div",X,[(0,r._)("div",B,[(0,r.Wm)(c,{to:"/",class:"logo"},{default:(0,r.w5)((()=>[J])),_:1}),(0,r._)("div",R,[(0,r.Wm)(c,{to:"/about",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)("文档")])),_:1}),(0,r.Wm)(c,{to:"/login",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)("登录")])),_:1}),(0,r.Wm)(c,{to:"/",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)("首页")])),_:1})]),(0,r._)("form",{onSubmit:t[5]||(t[5]=(0,s.iM)((function(){return e.onSubmit&&e.onSubmit(...arguments)}),["prevent"]))},[(0,r._)("div",Y,[(0,r.wy)((0,r._)("input",{type:"text","onUpdate:modelValue":t[0]||(t[0]=t=>e.username=t),placeholder:"输入用户名"},null,512),[[s.nr,e.username]]),(0,r.wy)((0,r._)("div",{class:"tip1"},(0,p.zw)(e.errors.username),513),[[s.F8,!e.validUsername]])]),(0,r._)("div",L,[(0,r.wy)((0,r._)("input",{type:"password","onUpdate:modelValue":t[1]||(t[1]=t=>e.password=t),placeholder:"输入登录密码"},null,512),[[s.nr,e.password]]),(0,r.wy)((0,r._)("div",{class:"tip2"},(0,p.zw)(e.errors.password),513),[[s.F8,!e.validPassword]])]),(0,r._)("div",Q,[(0,r.wy)((0,r._)("input",{type:"password","onUpdate:modelValue":t[2]||(t[2]=t=>e.confirmPassword=t),placeholder:"再次输入登录密码"},null,512),[[s.nr,e.confirmPassword]]),(0,r.wy)((0,r._)("div",{class:"tip3"},(0,p.zw)(e.errors.confirmPassword),513),[[s.F8,""!==e.password&&""!==e.confirmPassword&&e.password!==e.confirmPassword]])]),(0,r._)("div",E,[(0,r.wy)((0,r._)("input",{type:"text","onUpdate:modelValue":t[3]||(t[3]=t=>e.securityCode=t),placeholder:"输入验证码",class:"security-code-input"},null,512),[[s.nr,e.securityCode]]),(0,r._)("img",{src:e.securityCodeUrl,alt:"验证码",onClick:t[4]||(t[4]=function(){return e.refreshSecurityCode&&e.refreshSecurityCode(...arguments)}),class:"captcha-img"},null,8,Z),(0,r.wy)((0,r._)("div",{class:"tip4"},(0,p.zw)(e.errors.securityCode),513),[[s.F8,e.securityCodeInvalid]])]),K,(0,r._)("p",null,[(0,r.Uk)("已有账号？去"),(0,r.Wm)(c,{to:"/login",class:"routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)(" 登录")])),_:1})])],32)])])}var $=a(7016),ee=(0,r.aZ)({setup(){const e=(0,U.iH)(""),t=(0,U.iH)(""),a=(0,U.iH)(""),s=(0,U.iH)(""),n=(0,U.iH)(!1),l=new FormData,o={username:"",password:"",confirmPassword:"",securityCode:""},c=(0,U.iH)(""),i=(0,U.iH)(!1),d=(0,U.iH)(!1),u=(0,U.iH)(!1);(0,r.YP)(e,(async e=>{if(""===e.trim())return;const t=e.trim(),a=/^\w{6,20}$/.test(t);if(!a)return o.username="用户名为长度为6-20位的数字和字母的组合",void(i.value=!1);i.value=!0,o.username=""})),(0,r.YP)(t,(e=>{if(""===e.trim())return;const t=e;d.value=!!t&&/^[^\s]*(\s+[^\s]*){0}$/.test(t)&&/(?=.*[0-9])(?=.*[a-zA-Z])(?=.*[^a-zA-Z0-9])[^\s]{8,16}/.test(t),d.value?(d.value=!0,o.password=""):o.password="密码为8-16位数字、字母和特殊字符（不包括空格）"})),(0,r.YP)(a,(e=>{if(""===e.trim())return;const a=e.trim();u.value=!!a&&a===t.value})),(0,r.YP)([t,a],(e=>{let[t,a]=e;t&&a?u.value=!!t&&t===a:(u.value=!1,o.confirmPassword="两次输入的密码不一致！")}));const p=()=>{N().get("/user/refresh_code").then((e=>{c.value=e.data.image_url,n.value=!1,s.value=""})).catch((e=>{const t=e.response.data.error_msg;(0,$.bM)({type:"error",message:t})}))},m=()=>{i.value&&d.value&&u.value&&s.value&&(l.append("username",e.value),l.append("password",t.value),l.append("code",s.value),N().post("user/register/",l).then((e=>{console.log(e),Qt.push("/login")})).catch((e=>{if(e.response){const t=e.response.data.error_msg;(0,$.bM)({type:"warning",message:t})}})))};return p(),{username:e,password:t,confirmPassword:a,securityCode:s,errors:o,securityCodeUrl:c,validUsername:i,validPassword:d,validConfirmPassword:u,securityCodeInvalid:n,refreshSecurityCode:p,onSubmit:m}}});const te=(0,c.Z)(ee,[["render",G],["__scopeId","data-v-bbb8d43e"]]);var ae=te,se=a.p+"static/img/p1.88ab4c4f.png",re=a.p+"static/img/p2.62b0424d.png",ne=a.p+"static/img/p3.8ae5efd6.png",le=a.p+"static/img/p4.e2eea080.png",oe=a.p+"static/img/p5.1b2864c4.png",ce=a.p+"static/img/p6.a325ca95.png",ie=a.p+"static/img/p7.db9bfcc4.png",de=a.p+"static/img/p8.ed194e84.png",ue="data:image/png;base64,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",pe=a.p+"static/img/p10.05a69608.png",me=a.p+"static/img/p12.cc14bfe7.png",fe=a.p+"static/img/p13.15775013.png",ge=a.p+"static/img/p14.8d5201e2.png",ve=a.p+"static/img/p15.4537a7e8.png",we=a.p+"static/img/p16.7cee9a6b.png",he=a.p+"static/img/p17.e306acc6.png";const ye=e=>((0,r.dD)("data-v-112cce68"),e=e(),(0,r.Cn)(),e),be={class:"container"},_e={class:"body"},ke={style:{"text-indent":"2em"}},We=ye((()=>(0,r._)("p",{style:{"text-indent":"2em"}}," 在这里您可以在线使用自己的故障数据训练属于自己的故障分类模型，让模型更适合您的设备，故障分类更精准！在线训练得到的模型均支持下载，您也可以使用这些模型直接上传未分类数据进行分类预测，系统支持分类结果可视化和预测结果文件下载，预测结果更清晰！ ",-1))),Ue=ye((()=>(0,r._)("p",{style:{"text-indent":"2em"}},"下面开始您的专属模型训练和测试吧！",-1))),Ce=ye((()=>(0,r._)("p",{style:{"font-size":"12px","text-indent":"2.8em"}},"*为了显示效果最佳，请使用Edge、Chrome等浏览器",-1))),Ne=(0,r.uE)('<p style="text-indent:2em;" data-v-112cce68>如果您还没有账户请进行注册，以便您找到自己训练/测试的记录和得到的模型。</p><p style="text-indent:2em;" data-v-112cce68>1 点击首页注册键进入注册界面</p><img src="'+se+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>2 输入注册信息</p><p style="text-indent:2em;" data-v-112cce68>用户名：长度为6-20位的数字和字母的组合</p><p style="text-indent:2em;" data-v-112cce68>密码：8-16位数字、字母和特殊字符（不包括空格）</p><p style="text-indent:2em;" data-v-112cce68>输入验证码，若看不清可以点击切换</p><img src="'+re+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>3 注册成功会自动跳转进入登陆界面，输入用户名和密码即可</p><img src="'+ne+'" alt="" data-v-112cce68>',10),Ae=(0,r.uE)('<p style="text-indent:2em;" data-v-112cce68>1 点击首页登录键进入登录界面</p><img src="'+le+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>2 输入用户名和密码</p><img src="'+oe+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>3 登录成功即可进入主界面</p><img src="'+ce+'" alt="" data-v-112cce68>',6),Pe=(0,r.uE)('<p style="text-indent:2em;" data-v-112cce68>登录后直接进入系统功能界面，也可以通过首页创建项目键进入。</p><img src="'+ie+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>1 通过“选择文件”键上传您的训练数据集</p><img src="'+de+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>2 系统提供三种算法SVM（支持向量机）、RandomForest（随机森林）和CNN（卷积神经网络），其中SVM有进行K折交叉验证和不进行K折交叉验证两种，所以共有四种选择。勾选想要使用的算法即可。</p><img src="'+ue+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>3 输入模型名称：训练结束后会生成模型文件，这里定义的是模型文件名。</p><p style="text-indent:2em;" data-v-112cce68> 4 输入训练集数据中的特征数</p><p style="text-indent:2em;" data-v-112cce68>5 输入训练集中的标签范围（及类别数）</p><p style="text-indent:2em;" data-v-112cce68> 6 确定训练即可看到“训练进程展示”中增加了一条记录，显示当前训练的训练状态（训练中/已完成）、训练准确率、MacroF1值等。</p><img src="'+pe+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68> 7 训练结束后可以点击“下载模型”下载训练得到的模型，到浏览器的默认存储位置</p>',12),De=(0,r.uE)('<p style="text-indent:2em;" data-v-112cce68>1选择模型预测进入预测模块</p><img src="'+me+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>2上传待分类故障数据文件方式同训练过程</p><p style="text-indent:2em;" data-v-112cce68>3点击选择模型下拉列表即可看到所有自己训练的模型</p><img src="'+fe+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>4选择模型后会显示该模型的相关信息，您可以根据这些信息来判断模型是否适合当前上传的待分类数据。</p><img src="'+ge+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>5点击“确定创建”则测试结果栏中增加一条预测记录，展示“预测状态”、开始预测时间信息。</p><img src="'+ve+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>6预测结束后可以点击“可视化显示”键进入可视化界面。左侧为每个标签id对应的类别label值，右侧为每个类别的数据数量饼图。</p><img src="'+we+'" alt="" data-v-112cce68><p style="text-indent:2em;" data-v-112cce68>7在可视化显示界面中可以选择“下载结果”来下载每个标签及其对应label值的csv文件，选择“下载图形”则可以下载类别分布图。</p><img src="'+he+'" alt="" data-v-112cce68>',13);function xe(e,t){const a=(0,r.up)("router-link");return(0,r.wg)(),(0,r.iD)("div",be,[(0,r._)("div",_e,[(0,r._)("p",ke,[(0,r.Uk)(" 欢迎使用"),(0,r.Wm)(a,{to:"/",style:{"font-size":"16px",color:"#d85e1cc7","margin-left":"0px","font-weight":"600"}},{default:(0,r.w5)((()=>[(0,r.Uk)("分布式故障分类系统")])),_:1}),(0,r.Uk)("!")]),We,Ue,Ce,(0,r.Wm)(a,{to:"/register",style:{"font-weight":"600","text-indent":"2em",color:"#d85e1cc7"}},{default:(0,r.w5)((()=>[(0,r.Uk)("账户注册")])),_:1}),Ne,(0,r.Wm)(a,{to:"/login",style:{"font-weight":"600","text-indent":"2em",color:"#d85e1cc7"}},{default:(0,r.w5)((()=>[(0,r.Uk)("登录")])),_:1}),Ae,(0,r.Wm)(a,{to:"/TrainModel",style:{"text-indent":"2em","font-weight":"600",color:"#d85e1cc7"}},{default:(0,r.w5)((()=>[(0,r.Uk)("训练")])),_:1}),Pe,(0,r.Wm)(a,{to:"/ModelPrediction",style:{"text-indent":"2em","font-weight":"600",color:"#d85e1cc7"}},{default:(0,r.w5)((()=>[(0,r.Uk)("测试")])),_:1}),De])])}const Me={},Se=(0,c.Z)(Me,[["render",xe],["__scopeId","data-v-112cce68"]]);var Ve=Se;const Te=e=>((0,r.dD)("data-v-3e479e8a"),e=e(),(0,r.Cn)(),e),je={class:"WorkPage"},qe=Te((()=>(0,r._)("img",{src:m,alt:""},null,-1))),Fe={class:"username"},ze=Te((()=>(0,r._)("div",{class:"intomain"},null,-1)));function Ie(e,t,a,s,n,l){const o=(0,r.up)("router-link"),c=(0,r.up)("el-header"),i=(0,r.up)("el-aside"),d=(0,r.up)("router-view"),u=(0,r.up)("el-main"),m=(0,r.up)("el-container");return(0,r.wg)(),(0,r.iD)("div",je,[(0,r.Wm)(m,null,{default:(0,r.w5)((()=>[(0,r.Wm)(c,null,{default:(0,r.w5)((()=>[(0,r.Wm)(o,{to:"/",class:"logo routerlink"},{default:(0,r.w5)((()=>[qe])),_:1}),(0,r.Wm)(o,{to:"/about",class:"about routerlink"},{default:(0,r.w5)((()=>[(0,r.Uk)("使用文档")])),_:1}),(0,r._)("span",Fe,[(0,r.Uk)("欢迎，"),(0,r._)("span",{onClick:t[0]||(t[0]=function(){return e.handleLogout&&e.handleLogout(...arguments)}),class:"logout"},(0,p.zw)(e.username),1)])])),_:1}),(0,r.Wm)(m,null,{default:(0,r.w5)((()=>[(0,r.Wm)(i,{width:"15%"},{default:(0,r.w5)((()=>[(0,r._)("div",{onClick:t[1]||(t[1]=function(){return e.totrain&&e.totrain(...arguments)}),"exact-active-class":"active","active-class":"active"},"训练中心"),(0,r._)("div",{onClick:t[2]||(t[2]=function(){return e.totest&&e.totest(...arguments)}),"exact-active-class":"active","active-class":"active"},"模型预测")])),_:1}),(0,r.Wm)(u,null,{default:(0,r.w5)((()=>[ze,(0,r.Wm)(d)])),_:1})])),_:1})])),_:1})])}a(2801);const He=N().create({});He.interceptors.request.use((e=>{const t=localStorage.getItem("token");return t&&(e.headers.Authorization=`Token ${t}`),e}),(e=>Promise.reject(e)));N().create({});const{CancelToken:Oe}=N();async function Xe(){try{return await He.post("user/logout/"),localStorage.removeItem("token"),delete He.defaults.headers.common.Authorization,localStorage.removeItem("username"),!0}catch(e){return console.error(e),!1}}var Be=(0,r.aZ)({setup(){const e=localStorage.getItem("username"),t=async()=>{const e=await Xe();e?Qt.push("./login"):alert("退出失败！请稍后重试！")},a=()=>{Qt.push("/TrainModel")},s=()=>{Qt.push("/ModelPrediction")};return{username:e,handleLogout:t,totrain:a,totest:s}}});const Je=(0,c.Z)(Be,[["render",Ie],["__scopeId","data-v-3e479e8a"]]);var Re=Je;const Ye=e=>((0,r.dD)("data-v-32288e0a"),e=e(),(0,r.Cn)(),e),Le=Ye((()=>(0,r._)("h1",null,"404 Not Found!",-1)));function Qe(e,t,a,s,n,l){const o=(0,r.up)("router-link");return(0,r.wg)(),(0,r.iD)("div",null,[Le,(0,r._)("p",null,[(0,r.Uk)((0,p.zw)(e.count)+"秒后将自动跳转至",1),(0,r.Wm)(o,{to:"/"},{default:(0,r.w5)((()=>[(0,r.Uk)("首页")])),_:1})])])}var Ee=(0,r.aZ)({setup(){let e=(0,U.iH)(5);const t=setInterval((()=>{e.value--}),1e3);return(0,r.YP)(e,(e=>{0===e&&(clearInterval(t),Qt.push("/"))})),{count:e}}});const Ze=(0,c.Z)(Ee,[["render",Qe],["__scopeId","data-v-32288e0a"]]);var Ke=Ze;const Ge=e=>((0,r.dD)("data-v-cff8101e"),e=e(),(0,r.Cn)(),e),$e={class:"container"},et=Ge((()=>(0,r._)("h3",null,"创建新的训练",-1))),tt={class:"data"},at=Ge((()=>(0,r._)("div",{class:"el-upload__tip"},"支持 .csv 格式文件",-1))),st=Ge((()=>(0,r._)("h3",{style:{"margin-top":"20px","margin-bottom":"10px"}},"训练进程展示",-1))),rt={key:0},nt={key:1},lt={key:2},ot={key:3},ct={key:4},it={key:0},dt={key:0};function ut(e,t,a,s,n,l){const o=(0,r.up)("el-button"),c=(0,r.up)("el-upload"),i=(0,r.up)("el-form-item"),d=(0,r.up)("el-form"),u=(0,r.up)("el-radio"),m=(0,r.up)("el-input"),f=(0,r.up)("el-table-column"),g=(0,r.up)("el-table");return(0,r.wg)(),(0,r.iD)("div",$e,[et,(0,r._)("div",tt,[(0,r.Wm)(d,null,{default:(0,r.w5)((()=>[(0,r.Wm)(i,{label:"上传文件"},{default:(0,r.w5)((()=>[(0,r.Wm)(c,{ref:"fileUpload","on-change":s.handleFileChange,"show-file-list":!1,"auto-upload":!1,accept:".csv",multiple:!1},{default:(0,r.w5)((()=>[(0,r.Wm)(o,{size:"small",type:"primary",class:"choseFile"},{default:(0,r.w5)((()=>[(0,r.Uk)("选择文件")])),_:1}),at])),_:1},8,["on-change"])])),_:1})])),_:1}),(0,r.Wm)(d,{model:s.form,ref:"createForm",style:{"max-width":"600px"}},{default:(0,r.w5)((()=>[(0,r.Wm)(i,{label:"选择算法"},{default:(0,r.w5)((()=>[((0,r.wg)(!0),(0,r.iD)(r.HY,null,(0,r.Ko)(s.models,(e=>((0,r.wg)(),(0,r.j4)(u,{modelValue:s.form.modelName,"onUpdate:modelValue":t[0]||(t[0]=e=>s.form.modelName=e),label:e.name,key:e.name},{default:(0,r.w5)((()=>[(0,r.Uk)((0,p.zw)(e.name),1)])),_:2},1032,["modelValue","label"])))),128))])),_:1}),(0,r.Wm)(i,{label:"模型名称"},{default:(0,r.w5)((()=>[(0,r.Wm)(m,{modelValue:s.form.userModelName,"onUpdate:modelValue":t[1]||(t[1]=e=>s.form.userModelName=e),placeholder:"请输入模型名称",onBlur:s.checkModelName},null,8,["modelValue","onBlur"])])),_:1}),(0,r.Wm)(i,{label:"特征值数"},{default:(0,r.w5)((()=>[(0,r.Wm)(m,{type:"number",modelValue:s.form.featuresNum,"onUpdate:modelValue":t[2]||(t[2]=e=>s.form.featuresNum=e),placeholder:"请输入特征值数"},null,8,["modelValue"])])),_:1}),(0,r.Wm)(i,{label:"标签范围"},{default:(0,r.w5)((()=>[(0,r.Wm)(m,{type:"number",modelValue:s.form.labelsNum,"onUpdate:modelValue":t[3]||(t[3]=e=>s.form.labelsNum=e),placeholder:"请输入标签范围"},null,8,["modelValue"])])),_:1}),(0,r.Wm)(i,null,{default:(0,r.w5)((()=>[(0,r.Wm)(o,{type:"primary",onClick:s.createTrain},{default:(0,r.w5)((()=>[(0,r.Uk)("确定创建")])),_:1},8,["onClick"]),(0,r.Wm)(o,{onClick:s.cancel},{default:(0,r.w5)((()=>[(0,r.Uk)("取消创建")])),_:1},8,["onClick"])])),_:1})])),_:1},8,["model"])]),st,(0,r.Wm)(g,{data:s.trainTasks,style:{"max-width":"1024px"},"header-cell-style":{backgroundColor:"#f5f7fa"}},{default:(0,r.w5)((()=>[(0,r.Wm)(f,{prop:"name",label:"训练任务"}),(0,r.Wm)(f,{prop:"status",label:"训练状态"},{default:(0,r.w5)((e=>{let{row:t}=e;return[""===t.status?((0,r.wg)(),(0,r.iD)("span",rt,"等待中")):"in_progress"===t.status?((0,r.wg)(),(0,r.iD)("span",nt,"训练中")):"terminated"===t.status?((0,r.wg)(),(0,r.iD)("span",lt,"已取消")):"exception"===t.status?((0,r.wg)(),(0,r.iD)("span",ot,"训练失败")):"success"===t.status?((0,r.wg)(),(0,r.iD)("span",ct,"已完成")):(0,r.kq)("",!0)]})),_:1}),(0,r.Wm)(f,{prop:"startTime",label:"开始时间"}),(0,r.Wm)(f,{label:"Accuracy"},{default:(0,r.w5)((e=>{let{row:t}=e;return["success"===t.status?((0,r.wg)(),(0,r.iD)("span",it,(0,p.zw)(t.accuracy),1)):(0,r.kq)("",!0)]})),_:1}),(0,r.Wm)(f,{label:"MacroF1"},{default:(0,r.w5)((e=>{let{row:t}=e;return["success"===t.status?((0,r.wg)(),(0,r.iD)("span",dt,(0,p.zw)(t.MacroF1),1)):(0,r.kq)("",!0)]})),_:1}),(0,r.Wm)(f,{label:"操作"},{default:(0,r.w5)((e=>{let{row:t}=e;return["success"===t.status?((0,r.wg)(),(0,r.j4)(o,{key:0,type:"primary",href:s.modelURL,onClick:e=>s.downloadModel(t.id)},{default:(0,r.w5)((()=>[(0,r.Uk)("下载模型")])),_:2},1032,["href","onClick"])):(0,r.kq)("",!0)]})),_:1})])),_:1},8,["data"])])}a(2062);var pt={name:"TrainingPage",setup(){const e=(0,U.qj)([{name:"SVM"},{name:"RandomForest_K"},{name:"CNN"},{name:"SVM_K"}]);let t=new FormData,a=(0,U.qj)([]),s=(0,U.qj)({modelName:"",userModelName:"",featuresNum:"",labelsNum:""}),r=null,n=(0,U.iH)(null);const l=e=>{t.delete("file"),r=e.name,t.append("file",e.raw,e.name),console.log(e.raw)},o=()=>{if(""===s.userModelName)(0,$.bM)({type:"error",message:"模型名不能为空"});else if(!/^\w{6,20}$/.test(s.userModelName))return void(0,$.bM)({type:"error",message:"模型名称由6-20位字母数字组成"})},c=()=>{const e=new Date,t=String(e.getMonth()+1).padStart(2,"0"),a=String(e.getDate()).padStart(2,"0"),s=String(e.getHours()).padStart(2,"0"),r=String(e.getMinutes()).padStart(2,"0"),n=`${t}-${a} ${s}:${r}`;return n.toString()},i=()=>{if(""===s.modelName||""===s.userModelName||""===s.featuresNum||""===s.labelsNum||!r)return void(0,$.bM)({type:"error",message:"请填写完整的训练任务信息"});a.unshift((0,U.qj)({id:null,name:s.userModelName,status:"in_progress",startTime:c(),accuracy:0,MacroF1:0})),console.log(a);const e=c();t.append("create_time",e),t.append("algorithm",s.modelName),t.append("model_name",s.userModelName),t.append("featuresNum",s.featuresNum),t.append("labelsNum",s.labelsNum),console.log(t.get("file")),He.post("upload/train/",t,{headers:{"Content-Type":"multipart/form-data"}}).then((e=>{d(),console.log("--------------------------","yes,train了"),(0,$.bM)({type:"success",message:"创建训练任务成功"});const t=e.data.data,a=(0,U.qj)({id:t.id,name:t.model_name,status:t.status,startTime:t.create_time,accuracy:0,MacroF1:0});this.trainTasks.push(a),s.userModelName="",s.featuresNum="",s.labelsNum="",s.modelName="",r="";const{fileUpload:n}=this.$refs;n&&n.clearFiles()})).catch((e=>{const t=e.response.status;if(401===t)(0,$.bM)({type:"error",message:"未校验，请重新登录！"}),Qt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,$.bM)({type:"error",message:t}),console.error(e)}}})).finally((()=>(u(),t.delete("name"),t.delete("file"),t.delete("algorithm"),t.delete("featuresNum"),t.delete("labelsNum"),r="",new Promise((()=>{})))))},d=()=>{He.get("upload/find_train/").then((e=>{const t=e.data.data;a.splice(0),t.forEach((e=>{a.push((0,U.qj)({id:e.train_id,name:e.model_name,status:e.status,startTime:e.create_time,accuracy:e.Accuracy,MacroF1:e.MacroF1}))})),a.reverse()})).catch((e=>{const t=e.response.status;if(401===t)(0,$.bM)({type:"error",message:"未校验，请重新登录！"}),Qt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,$.bM)({type:"error",message:t}),console.error(e)}}}))},u=()=>{if(s.userModelName="",s.featuresNum="",s.labelsNum="",s.modelName="",r="",t.delete("file"),n.value){const{fileUpload:e}=n.value;e&&e.clearFiles()}},p=(0,U.iH)(""),m=e=>{const t=new FormData;t.append("taskid",e),He.post("/upload/model_download/",t,{responseType:"blob"}).then((e=>{const t=e.headers["content-disposition"].match(/filename="([^"]+)"/)[1].replace(/"/g,""),a=URL.createObjectURL(e.data),s=document.createElement("a");s.style.display="none",s.href=a,s.setAttribute("download",t),document.body.appendChild(s),s.click(),URL.revokeObjectURL(a),document.body.removeChild(s)})).catch((e=>{const t=e.response.status;if(401===t)(0,$.bM)({type:"error",message:"未校验，请重新登录！"}),Qt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,$.bM)({type:"error",message:t}),console.error(e)}}}))};return d(),{models:e,formData:t,downloadModel:m,modelURL:p,trainTasks:a,form:s,fileUploadRef:n,handleFileChange:l,createTrain:i,getTrainTasks:d,cancel:u,checkModelName:o}}};const mt=(0,c.Z)(pt,[["render",ut],["__scopeId","data-v-cff8101e"]]);var ft=mt;const gt=e=>((0,r.dD)("data-v-3abe2739"),e=e(),(0,r.Cn)(),e),vt={class:"container"},wt=gt((()=>(0,r._)("h3",null,"开始预测",-1))),ht={class:"data"},yt=gt((()=>(0,r._)("div",{class:"el-upload__tip"},"支持 .csv 格式文件",-1))),bt={class:"modelCard"},_t={class:"choseModel"},kt=gt((()=>(0,r._)("span",{style:{"font-size":"14px",color:"var(--el-text-color-regular)","margin-right":"12px"}},"选择模型",-1))),Wt={key:0,class:"showModel"},Ut={class:"card-item"},Ct={class:"card-content-item"},Nt=gt((()=>(0,r._)("h3",{style:{"margin-top":"20px","margin-bottom":"10px"}},"测试结果",-1))),At={key:0},Pt={key:1},Dt={key:2},xt={key:3},Mt={key:4};function St(e,t,a,s,n,l){const o=(0,r.up)("el-button"),c=(0,r.up)("el-upload"),i=(0,r.up)("el-form-item"),d=(0,r.up)("el-form"),u=(0,r.up)("el-option"),m=(0,r.up)("el-select"),f=(0,r.up)("el-card"),g=(0,r.up)("el-input"),v=(0,r.up)("el-table-column"),w=(0,r.up)("el-table");return(0,r.wg)(),(0,r.iD)("div",vt,[wt,(0,r._)("div",ht,[(0,r.Wm)(d,null,{default:(0,r.w5)((()=>[(0,r.Wm)(i,{label:"上传文件"},{default:(0,r.w5)((()=>[(0,r.Wm)(c,{ref:"fileUpload","on-change":s.handleFileChange,"show-file-list":!1,"auto-upload":!1,accept:".csv",multiple:!1},{default:(0,r.w5)((()=>[(0,r.Wm)(o,{size:"small",type:"primary",class:"choseFile"},{default:(0,r.w5)((()=>[(0,r.Uk)("选择文件")])),_:1}),yt])),_:1},8,["on-change"])])),_:1})])),_:1}),(0,r._)("div",bt,[(0,r._)("div",_t,[kt,(0,r.Wm)(m,{label:"选择模型",modelValue:s.form.modelid,"onUpdate:modelValue":t[0]||(t[0]=e=>s.form.modelid=e),onChange:t[1]||(t[1]=e=>s.handleModelChange(s.form.modelid))},{default:(0,r.w5)((()=>[((0,r.wg)(!0),(0,r.iD)(r.HY,null,(0,r.Ko)(s.models,(e=>((0,r.wg)(),(0,r.j4)(u,{value:e.id,label:e.name,key:e.id},{default:(0,r.w5)((()=>[(0,r.Uk)((0,p.zw)(e.name),1)])),_:2},1032,["value","label"])))),128))])),_:1},8,["modelValue"])]),0!==Object.keys(s.selectedModel).length?((0,r.wg)(),(0,r.iD)("div",Wt,[(0,r.Wm)(f,{shadow:"hover"},{default:(0,r.w5)((()=>[(0,r._)("div",Ut,[(0,r._)("div",null,"模型名称："+(0,p.zw)(s.selectedModel.name),1),(0,r._)("div",Ct,"特征值数: "+(0,p.zw)(s.selectedModel.featuresNum),1)])])),_:1})])):(0,r.kq)("",!0)]),(0,r.Wm)(d,null,{default:(0,r.w5)((()=>[(0,r.Wm)(i,{label:"特征值数"},{default:(0,r.w5)((()=>[(0,r.Wm)(g,{type:"number",modelValue:s.form.featuresNum,"onUpdate:modelValue":t[2]||(t[2]=e=>s.form.featuresNum=e),placeholder:"请输入特征值数",style:{width:"215px"}},null,8,["modelValue"])])),_:1}),(0,r.Wm)(i,null,{default:(0,r.w5)((()=>[(0,r.Wm)(o,{type:"primary",onClick:s.createPrediction},{default:(0,r.w5)((()=>[(0,r.Uk)("确定创建")])),_:1},8,["onClick"]),(0,r.Wm)(o,{onClick:s.cancel},{default:(0,r.w5)((()=>[(0,r.Uk)("取消创建")])),_:1},8,["onClick"])])),_:1})])),_:1})]),Nt,(0,r.Wm)(w,{data:s.predictionTasks,style:{"max-width":"1024px"},"header-cell-style":{backgroundColor:"#f5f7fa"}},{default:(0,r.w5)((()=>[(0,r.Wm)(v,{prop:"name",label:"预测任务"}),(0,r.Wm)(v,{prop:"status",label:"预测状态"},{default:(0,r.w5)((e=>{let{row:t}=e;return[""===t.status?((0,r.wg)(),(0,r.iD)("span",At,"等待中")):"in_progress"===t.status?((0,r.wg)(),(0,r.iD)("span",Pt,"预测中")):"terminated"===t.status?((0,r.wg)(),(0,r.iD)("span",Dt,"已取消")):"exception"===t.status?((0,r.wg)(),(0,r.iD)("span",xt,"预测失败")):"success"===t.status?((0,r.wg)(),(0,r.iD)("span",Mt,"已完成")):(0,r.kq)("",!0)]})),_:1}),(0,r.Wm)(v,{prop:"startTime",label:"开始时间"}),(0,r.Wm)(v,{label:"操作"},{default:(0,r.w5)((e=>{let{row:t}=e;return["success"===t.status?((0,r.wg)(),(0,r.j4)(o,{key:0,type:"primary",onClick:e=>s.showCharts(t.id)},{default:(0,r.w5)((()=>[(0,r.Uk)("可视化展示")])),_:2},1032,["onClick"])):(0,r.kq)("",!0)]})),_:1})])),_:1},8,["data"])])}var Vt={setup(){let e=(0,U.qj)([]),t=new FormData,a=(0,U.qj)([]),s=(0,U.qj)({modelid:"",featuresNum:""}),r=(0,U.iH)({});const n=()=>{He.get("upload/find_model/").then((t=>{const a=t.data.data;console.log(a),a.forEach((t=>{const a=(0,U.qj)({id:t.model_id,name:t.model_name,featuresNum:t.featuresNum});e.push(a)})),console.log("------------------------",e)})).catch((e=>{const t=e.response.status;if(401===t)(0,$.bM)({type:"error",message:"未校验，请重新登录！"}),Qt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,$.bM)({type:"error",message:t}),console.error(e)}}}))};function l(t){r.value=e.find((e=>e.id===t))}const o=e=>{Qt.push({path:`/ResultsPage/${e}`})};let c=null,i=(0,U.iH)(null);const d=e=>{t.delete("file"),c=e.name,t.append("file",e.raw,e.name),console.log(e.raw)},u=()=>{const e=new Date,t=String(e.getMonth()+1).padStart(2,"0"),a=String(e.getDate()).padStart(2,"0"),s=String(e.getHours()).padStart(2,"0"),r=String(e.getMinutes()).padStart(2,"0"),n=`${t}-${a} ${s}:${r}`;return n.toString()},p=()=>{if(""===s.modelid||""===s.featuresNum||!c)return void(0,$.bM)({type:"error",message:"请填写完整的训练任务信息"});t.append("model_id",s.modelid),t.append("featuresNum",s.featuresNum),console.log(t.get("file")),m(),a.unshift((0,U.qj)({id:null,name:c,status:"in_progress",startTime:u()}));const e=u();t.append("create_time",e),He.post("upload/test/",t,{headers:{"Content-Type":"multipart/form-data"}}).then((e=>{m(),(0,$.bM)({type:"success",message:"创建预测任务成功"});const t=e.data.data;a.push((0,U.qj)({id:t.id,name:c,status:"in_progress",startTime:t.create_time})),s.featuresNum="",s.modelid="",c="",r.value={}})).catch((e=>{const t=e.response.status;if(401===t)(0,$.bM)({type:"error",message:"未校验，请重新登录！"}),Qt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,$.bM)({type:"error",message:t}),console.error(e)}}})).finally((()=>(f(),t.delete("model_id"),t.delete("file"),t.delete("featuresNum"),c="",new Promise((()=>{})))))},m=()=>{He.get("upload/find_test/").then((e=>{a.splice(0);const t=e.data.data;t.forEach((e=>{a.push((0,U.qj)({id:e.test_id,name:e.name,status:e.status,startTime:e.create_time}))})),a.reverse()})).catch((e=>{const t=e.response.status;if(401===t)(0,$.bM)({type:"error",message:"未校验，请重新登录！"}),Qt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,$.bM)({type:"error",message:t}),console.error(e)}}}))},f=()=>{s.modelid="",s.featuresNum="",r.value={},c="",t.delete("file")};return n(),m(),{models:e,formData:t,predictionTasks:a,fileUploadRef:i,form:s,selectedModel:r,showCharts:o,handleFileChange:d,handleModelChange:l,createPrediction:p,getPredictionTasks:m,cancel:f}}};const Tt=(0,c.Z)(Vt,[["render",St],["__scopeId","data-v-3abe2739"]]);var jt=Tt;const qt={class:"container"},Ft={class:"download-btns"},zt={class:"show"},It={class:"table-container"},Ht={ref:"pieCanvas",class:"pieCanvas"};function Ot(e,t,a,s,n,l){const o=(0,r.up)("el-button"),c=(0,r.up)("el-table-column"),i=(0,r.up)("el-table");return(0,r.wg)(),(0,r.iD)("div",qt,[(0,r._)("div",Ft,[(0,r.Wm)(o,{type:"primary",onClick:l.downloadTableData},{default:(0,r.w5)((()=>[(0,r.Uk)(" 下载结果 ")])),_:1},8,["onClick"]),(0,r.Wm)(o,{type:"primary",onClick:l.downloadPieChart,style:{"margin-left":"65px"}},{default:(0,r.w5)((()=>[(0,r.Uk)(" 下载图形 ")])),_:1},8,["onClick"])]),(0,r._)("div",zt,[(0,r._)("div",It,[(0,r.Wm)(i,{data:n.tableData,class:"table"},{default:(0,r.w5)((()=>[(0,r.Wm)(c,{prop:"sampleId",label:"Sample Id"}),(0,r.Wm)(c,{prop:"label",label:"Label"})])),_:1},8,["data"])]),(0,r._)("canvas",Ht,null,512)])])}a(1439),a(7585),a(5315);var Xt=a(2305),Bt={name:"MyComponent",props:["taskId"],data(){return{tableData:[],pieChartData:[],pieChart:null}},mounted(){const e=new FormData;e.append("taskid",parseInt(this.taskId)),this.$nextTick((()=>{this.$refs.pieCanvas&&this.renderPieChart()})),He.post("/upload/result_download/",e).then((e=>{const t=e.data.data;let a=t[0],s=t[1];const r=[],n=[];for(let l in a)r.push({sampleId:l,label:a[l]});for(let l in s)n.push({label:l,value:s[l]});this.tableData=(0,U.qj)(r),this.pieChartData=(0,U.qj)(n),this.renderPieChart()})).catch((e=>{const t=e.response.status;if(401===t)(0,$.bM)({type:"error",message:"未校验，请重新登录！"}),Qt.push("/login");else{if(400!==t)return Promise.reject(e);{const t=e.response.data.error_msg;(0,$.bM)({type:"error",message:t}),console.error(e)}}}))},methods:{renderPieChart(){if(!this.$refs.pieCanvas)return;this.pieChart=Xt.S1(this.$refs.pieCanvas,null,{devicePixelRatio:2});const e=this.pieChartData.map((e=>({value:parseInt(e.value),name:e.label}))),t={grid:{width:"60%",height:"60%",left:"center"},series:[{type:"pie",data:e,label:{show:!0,formatter:"{b}: {c}",textStyle:{fontSize:14}},emphasis:{label:{show:!0,textStyle:{fontSize:1}}}}]};this.pieChart.setOption(t)},downloadTableData(){const e=this.arrayToCSV(this.tableData),t=new Blob([e],{type:"text/csv;charset=utf-8;"}),a=document.createElement("a");a.href=URL.createObjectURL(t),a.download="table-data.csv",a.click()},downloadPieChart(){if(!this.$refs.pieCanvas)return;const e=this.$refs.pieCanvas.toDataURL("image/png"),t=this.dataURItoBlob(e),a=document.createElement("a");a.href=URL.createObjectURL(t),a.download="pie-chart.png",a.click()},dataURItoBlob(e){const t=atob(e.split(",")[1]),a=[];for(let r=0;r<t.length;r++)a.push(t.charCodeAt(r));const s=e.split(",")[0].split(":")[1].split(";")[0];return new Blob([new Uint8Array(a)],{type:s})},arrayToCSV(e){const t=[],a=Object.keys(e[0]);t.push(a.join(","));for(const s of e){const e=a.map((e=>{const t=null===s[e]||void 0===s[e]?"":s[e];return JSON.stringify(t)}));t.push(e.join(","))}return t.join("\n")}}};const Jt=(0,c.Z)(Bt,[["render",Ot],["__scopeId","data-v-63633d4d"]]);var Rt=Jt;const Yt=[{path:"/:pathMatch(.*)*",component:Ke},{path:"/login",component:D},{path:"/",component:H},{path:"/register",component:ae},{path:"/about",component:Ve},{path:"/workspace",component:Re,children:[{path:"/TrainModel",component:ft},{path:"/ResultsPage/:taskId",component:Rt,props:!0},{path:"/ModelPrediction",component:jt}]}],Lt=(0,u.p7)({history:(0,u.PO)("/"),routes:Yt});var Qt=Lt,Et=a(502);a(2834);const Zt=localStorage.getItem("token");He.defaults.headers.common.Authorization=Zt?`Bearer ${Zt}`:"",(0,s.ri)(d).use(Qt).use(Et.Z).mount("#app")}},t={};function a(s){var r=t[s];if(void 0!==r)return r.exports;var n=t[s]={exports:{}};return e[s].call(n.exports,n,n.exports,a),n.exports}a.m=e,function(){var e=[];a.O=function(t,s,r,n){if(!s){var l=1/0;for(d=0;d<e.length;d++){s=e[d][0],r=e[d][1],n=e[d][2];for(var o=!0,c=0;c<s.length;c++)(!1&n||l>=n)&&Object.keys(a.O).every((function(e){return a.O[e](s[c])}))?s.splice(c--,1):(o=!1,n<l&&(l=n));if(o){e.splice(d--,1);var i=r();void 0!==i&&(t=i)}}return t}n=n||0;for(var d=e.length;d>0&&e[d-1][2]>n;d--)e[d]=e[d-1];e[d]=[s,r,n]}}(),function(){a.n=function(e){var t=e&&e.__esModule?function(){return e["default"]}:function(){return e};return a.d(t,{a:t}),t}}(),function(){a.d=function(e,t){for(var s in t)a.o(t,s)&&!a.o(e,s)&&Object.defineProperty(e,s,{enumerable:!0,get:t[s]})}}(),function(){a.g=function(){if("object"===typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"===typeof window)return window}}()}(),function(){a.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)}}(),function(){a.r=function(e){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})}}(),function(){a.p="/"}(),function(){var e={143:0};a.O.j=function(t){return 0===e[t]};var t=function(t,s){var r,n,l=s[0],o=s[1],c=s[2],i=0;if(l.some((function(t){return 0!==e[t]}))){for(r in o)a.o(o,r)&&(a.m[r]=o[r]);if(c)var d=c(a)}for(t&&t(s);i<l.length;i++)n=l[i],a.o(e,n)&&e[n]&&e[n][0](),e[n]=0;return a.O(d)},s=self["webpackChunkfaultc"]=self["webpackChunkfaultc"]||[];s.forEach(t.bind(null,0)),s.push=t.bind(null,s.push.bind(s))}();var s=a.O(void 0,[998],(function(){return a(7406)}));s=a.O(s)})();
//# sourceMappingURL=app.a9ea6292.js.map