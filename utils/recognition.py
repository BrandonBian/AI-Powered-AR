# Link: https://github.com/clovaai/deep-text-recognition-benchmark

def demo(opt, roi):

    predict_list = []

    with torch.no_grad():
        batch_size = roi.size(0)
        image = roi.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index, preds_size)

        else:
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        # log = open(f'./log_demo_result.txt', 'a')
        dashed_line = '-' * 80

        head = f'{"predicted_labels":25s}\tconfidence score\t'

        # print(f'{dashed_line}\n{head}\n{dashed_line}')
        # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            # print(f'\t{pred:25s}\t{confidence_score:0.4f}')

            predict_list.append(pred)
        return (predict_list)
