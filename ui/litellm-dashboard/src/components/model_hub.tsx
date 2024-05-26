import React, { useEffect, useState } from 'react';

import { modelHubCall } from "./networking";

import { Card, Text, Title, Grid, Button } from "@tremor/react";

import { RightOutlined } from '@ant-design/icons';

import { Modal } from 'antd';


interface ModelHubProps {

  userID: string;

  userRole: string;

  token: string;

  accessToken: string;

  keys: any; // Replace with the appropriate type for 'keys' prop

  premiumUser: boolean;

}



const ModelHub: React.FC<ModelHubProps> = ({

  userID,

  userRole,

  token,

  accessToken,

  keys,

  premiumUser,

}) => {

  const [modelHubData, setModelHubData] = useState(null);

  const [isModalVisible, setIsModalVisible] = useState(false);

  const [selectedModel, setSelectedModel] = useState(null);


  useEffect(() => {

    if (!accessToken || !token || !userRole || !userID) {

      return;

    }



    const fetchData = async () => {

      try {

        const _modelHubData = await modelHubCall(accessToken, userID, userRole);

        console.log("ModelHubData:", _modelHubData);

        setModelHubData(_modelHubData.data);

        

      } catch (error) {

        console.error("There was an error fetching the model data", error);

      }

    };



    fetchData();

  }, [accessToken, token, userRole, userID]);



  const showModal = (model) => {

    setSelectedModel(model);

    setIsModalVisible(true);

  };



  const handleOk = () => {

    setIsModalVisible(false);

    setSelectedModel(null);

  };



  const handleCancel = () => {

    setIsModalVisible(false);

    setSelectedModel(null);

  };



  return (

    <div>

      <Title>Model Hub</Title>

      <div style={{ width: '100%' }}>

      <div className="grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-4">

          {modelHubData && modelHubData.map((model: any) => (
    <Card

    key={model.model_id}

    className="mt-5 mx-8"

  >

    <pre>

    <Title level={4}>{model.model_name}</Title>

    </pre>

    <div style={{ marginTop: 'auto', textAlign: 'right' }}>

      <a href="#" onClick={() => showModal(model)} style={{ color: '#1890ff', fontSize: 'smaller' }}>

        View more <RightOutlined />

      </a>

    </div>

  </Card>

          ))}

        </div>

      </div>

      <Modal

        title="Model Usage"

        visible={isModalVisible}

        onOk={handleOk}

        onCancel={handleCancel}

      >

        {selectedModel && (

          <div>

            <p><strong>Model Name:</strong> {selectedModel.model_name}</p>

            <p><strong>Additional Params:</strong> {JSON.stringify(selectedModel.litellm_params)}</p>

            {/* Add other model details here */}

          </div>

        )}

      </Modal>

    </div>

  );

};



export default ModelHub;